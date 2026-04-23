// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// -convert-hc-front-to-hc: mechanical structural rewrite from hc_front to hc.
//
// The `hc_front` dialect is the source-faithful AST serialization; every
// value is `!hc_front.value` and names/attributes/calls are left unresolved.
// The Python driver (see `hc/_resolve.py`) stamps every `hc_front.name` and
// `hc_front.attr` with a `ref` DictAttr classifying the symbol. This pass
// consumes that classification: for every `hc_front` top-level callable it
// walks the body once and emits the parallel `hc` callable alongside, then
// erases the source op.
//
// Scope for this bead (5-1fh):
//  * the structural rewrites for every `hc_front` op the WMMA kernel touches
//    in v0 (kernel/func/intrinsic, workitem/subgroup regions, for-range over
//    a `range(...)` iter, constant, binop, name dispatched on `ref`,
//    target_* + assign, slice, tuple/keyword glue, subscript, call dispatched
//    on the callee's `ref`, plus a small `dsl_method` subset that's mechanical
//    enough to fit — `a.shape[N]`, `x.vec()`, `x.with_inactive(value=...)`,
//    `x.astype(...)`);
//  * every produced value gets `!hc.undef` — type inference pins later.
//
// Explicitly out of scope (separate beads):
//  * loop-carried iter_arg analysis and nested workitem-def folding (5-3jb).
//  * intrinsic body discarding (5-uv5).
//  * full inlining of `ref.kind = "inline"` helpers; for now we leave the
//    `hc_front.call` pinned with a diagnostic if it survives lowering.

#include "hc/Conversion/HCFrontToHC/HCFrontToHC.h"

#include "hc/Front/IR/HCFrontDialect.h"
#include "hc/Front/IR/HCFrontOps.h"
#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace mlir::hc::front {
#define GEN_PASS_DEF_CONVERTHCFRONTTOHC
#include "hc/Conversion/HCFrontToHC/Passes.h.inc"
} // namespace mlir::hc::front

using namespace mlir;
using namespace mlir::hc;
namespace hc_front = mlir::hc::front;

namespace {

// The Python driver emits launch-geo axes as small non-negative ints
// (0..<launch_rank). We don't know launch rank at this site, so reject
// anything that would build a pathologically large variadic result list
// up front — a hand-crafted IR with `axis = -1` or `axis = 2^31` should
// diagnose, not allocate gigabytes of `!hc.undef` result types.
constexpr int64_t kMaxLaunchAxis = 32;

//===----------------------------------------------------------------------===//
// Scope map. Each callable / region owns a string -> Value scope that the
// `hc_front.name` lookup consults. Scopes chain through their parent so an
// inner `hc_front.workitem_region` sees the enclosing callable's params and
// locals.
//===----------------------------------------------------------------------===//

struct Scope {
  Scope *parent = nullptr;
  llvm::StringMap<Value> bindings;

  Value lookup(StringRef name) const {
    for (const Scope *s = this; s; s = s->parent) {
      auto it = s->bindings.find(name);
      if (it != s->bindings.end())
        return it->second;
    }
    return nullptr;
  }

  void bind(StringRef name, Value v) { bindings[name] = v; }
};

//===----------------------------------------------------------------------===//
// The Python driver (see `hc/_resolve.py`) stamps every `hc_front.name` and
// most `hc_front.attr` with a `ref` DictAttr of the form
//   {kind = "<class>", ...per-class payload...}
// `RefInfo` is the null-safe, minimally-typed view: it's cheap to construct,
// survives a missing `ref` on hand-written IR, and centralizes attribute
// lookups so every call site goes through the same code path.
//===----------------------------------------------------------------------===//

struct RefInfo {
  DictionaryAttr dict;
  StringRef kind;

  static RefInfo of(Operation *op) {
    RefInfo info;
    if (!op)
      return info;
    info.dict = op->getAttrOfType<DictionaryAttr>("ref");
    if (info.dict) {
      if (auto k = info.dict.getAs<StringAttr>("kind"))
        info.kind = k.getValue();
    }
    return info;
  }

  explicit operator bool() const { return static_cast<bool>(dict); }

  StringRef string(StringRef key) const {
    if (!dict)
      return {};
    if (auto s = dict.getAs<StringAttr>(key))
      return s.getValue();
    return {};
  }

  template <typename AttrT> AttrT getAs(StringRef key) const {
    if (!dict)
      return {};
    return dict.getAs<AttrT>(key);
  }
};

//===----------------------------------------------------------------------===//
// Attribute-conversion helpers.
//===----------------------------------------------------------------------===//

// Turn a plain builtin `["M + 1", ...]` string-array attribute (the form
// `hc_front` uses for `work_shape` / `group_shape`) into a `#hc.shape<...>`
// attribute. Returns null on a malformed dimension so the caller can emit a
// diagnostic against the source op.
FailureOr<ShapeAttr> stringArrayToShape(MLIRContext *ctx, Operation *sourceOp,
                                        ArrayAttr array) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallVector<Attribute> dims;
  dims.reserve(array.size());
  for (Attribute item : array) {
    auto str = dyn_cast<StringAttr>(item);
    if (!str) {
      sourceOp->emitOpError(
          "expected string entry in shape-like attribute, got ")
          << item;
      return failure();
    }
    std::string diag;
    FailureOr<sym::ExprHandle> handle =
        sym::parseExpr(store, str.getValue(), &diag);
    if (failed(handle)) {
      sourceOp->emitOpError("failed to parse hc.shape dim '")
          << str.getValue() << "': " << diag;
      return failure();
    }
    dims.push_back(ExprAttr::get(ctx, *handle));
  }
  return ShapeAttr::get(ctx, dims);
}

// The front-end emits `effects = "pure"` / `"read"` / `"write"` /
// `"read_write"` as a plain string. Translate into the typed `HC_EffectsAttr`
// that `hc.func` / `hc.intrinsic` want.
std::optional<EffectClass> parseEffectClass(StringRef text) {
  return llvm::StringSwitch<std::optional<EffectClass>>(text)
      .Case("pure", EffectClass::Pure)
      .Case("read", EffectClass::Read)
      .Case("write", EffectClass::Write)
      .Case("read_write", EffectClass::ReadWrite)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// Binary-op mnemonic table. `hc_front.binop "Add"(a, b)` spells out the
// Python AST's BinOp.op class name; the lowering picks the matching `hc`
// op and emits it against the two already-lowered operands. All results
// use `!hc.undef`, consistent with the rest of the pipeline's progressive
// typing.
//===----------------------------------------------------------------------===//

Value emitBinop(OpBuilder &builder, Location loc, StringRef kind, Value lhs,
                Value rhs, Type undef, Operation *sourceOp) {
  if (kind == "Add")
    return HCAddOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Sub")
    return HCSubOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Mult")
    return HCMulOp::create(builder, loc, undef, lhs, rhs);
  // `hc.div` is documented as "integer/float division (Python `//` for
  // ints)", i.e. a single typed division whose semantics are pinned at
  // type-inference time. Both Python `/` (Div) and `//` (FloorDiv) land
  // here; for integer operands both floor, for floats both are true
  // division. If a future bead splits truediv from floordiv, only this
  // branch needs to change.
  if (kind == "FloorDiv" || kind == "Div")
    return HCDivOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Mod")
    return HCModOp::create(builder, loc, undef, lhs, rhs);
  sourceOp->emitOpError("unsupported hc_front.binop kind '") << kind << "'";
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Lowerer. Instantiated once per top-level `hc_front` callable; walks the
// body, threads the scope map through nested regions, and emits the
// corresponding `hc` ops via the shared `builder`. Source ops are erased
// after each top-level op finishes lowering.
//===----------------------------------------------------------------------===//

class Lowerer {
public:
  Lowerer(OpBuilder &builder, Type undef) : builder(builder), undef(undef) {}

  LogicalResult lowerCallable(Operation *frontOp);

private:
  OpBuilder &builder;
  Type undef;

  // Stack of scope frames; the top frame matches the current innermost
  // region being lowered. `lookupName` walks the chain via `Scope::parent`.
  llvm::SmallVector<std::unique_ptr<Scope>> scopes;

  // `hc_front.tuple` produces a single SSA value, but the downstream ops
  // (assign-to-target_tuple, keyword-shape, variadic subscript) need to
  // destructure it back into its element list. Recording the lowered
  // operand list here avoids re-walking the tuple's source ops.
  llvm::DenseMap<Value, SmallVector<Value>> tupleElts;

  // `hc_front.keyword "name" = %v` is consumed at the parent call. The
  // map is keyed by the keyword op's SSA result and stores the name, the
  // original hc_front value (so shape-tuple folds can still reach into
  // `tupleElts`), and the lowered value (may be null for tuple/attr
  // operands that have no standalone hc counterpart).
  struct KeywordInfo {
    StringRef name;
    Value origValue;
    Value loweredValue;
  };
  llvm::DenseMap<Value, KeywordInfo> keywordInfo;

  // Every SSA value produced inside the `hc_front` body maps to the `hc`
  // value that replaces it. This is the workhorse translation table used
  // by every operand lookup.
  llvm::DenseMap<Value, Value> valueMap;

  Scope &currentScope() { return *scopes.back(); }

  // Resolves a classified `hc_front.name` against the current scope chain
  // and the `ref` DictAttr.
  // Returns:
  //   * `success(Value)`  — a usable hc value for the name;
  //   * `success(Value())` — the name is consumed at the call site and
  //                          intentionally has no SSA counterpart (e.g.
  //                          callee/intrinsic/inline/builtin/module refs);
  //   * `failure()`        — a diagnostic was emitted.
  FailureOr<Value> lowerName(hc_front::NameOp op);

  // Dispatches on `ref.kind = "dsl_method"`'s `method` payload, or falls
  // through to a module/attr chain the call site handles.
  Value lowerAttr(hc_front::AttrOp op);

  LogicalResult lowerRegion(Region &src);

  // Shared body for `hc.workitem_region` / `hc.subgroup_region`. Both have
  // identical shape (captures attr + single-block body + optional
  // parameters stamped on the front op that become block args) so we
  // route through one helper templated on the target `hc` op type.
  template <typename HCRegionOpT, typename FrontRegionOpT>
  LogicalResult lowerCapturingRegion(FrontRegionOpT op);

  LogicalResult lowerOp(Operation *op);

  Value lowerValueOperand(Value v);
  SmallVector<Value> lowerValueOperands(ValueRange vs);
  SmallVector<Value> expandTupleOperand(Value v);

  // Per-op lowering entry points. `Value`-returning variants write into
  // `valueMap`; `LogicalResult`-returning ones are terminators or
  // side-effecting ops without a user-visible result.
  Value lowerConstant(hc_front::ConstantOp op);
  Value lowerBinop(hc_front::BinOp op);
  Value lowerSlice(hc_front::SliceOp op);
  LogicalResult lowerReturn(hc_front::ReturnOp op);
  LogicalResult lowerAssign(hc_front::AssignOp op);
  LogicalResult lowerFor(hc_front::ForOp op);
  LogicalResult lowerWorkitemRegion(hc_front::WorkitemRegionOp op);
  LogicalResult lowerSubgroupRegion(hc_front::SubgroupRegionOp op);
  // FailureOr so a `builtin` call that is cleanly consumed by its parent
  // (`range(...)` inside `hc_front.for`) can be distinguished from a real
  // error. `FailureOr<Value>` reads as: success+Value / success+null /
  // failure.
  FailureOr<Value> lowerCall(hc_front::CallOp op);
  Value lowerSubscript(hc_front::SubscriptOp op);

  // Helpers for populating a fresh callable's entry block from an
  // `hc_front` `parameters = [...]` attribute. Binds the param names into
  // `scope` so subsequent `hc_front.name {ref.kind = "param"}` lookups
  // resolve to the right block argument.
  FailureOr<FunctionType> materializeParameters(ArrayAttr params, Block &entry,
                                                Scope &scope,
                                                Operation *sourceOp,
                                                bool returnsValue);

  // Emit an `hc.const` for a `ref.kind = "constant"` name op. Needs the
  // python_kind / value payload because the front dialect packs it as
  // string-for-anything-but-int.
  Value emitConstantFromRef(Location loc, const RefInfo &ref,
                            Operation *sourceOp);

  // Consume a `hc_front.call` dispatched to a `dsl_method` callee. The
  // FailureOr<Value> return mirrors `lowerCall`: failure = error emitted,
  // success+null = the call produced no SSA result (e.g. `x.store(...)`).
  FailureOr<Value> lowerDslMethodCall(hc_front::CallOp call,
                                      hc_front::AttrOp attr);

  // Emit the launch-geometry op for a `group.{method}` DSL attribute, if
  // `method` names one. `resultIdx` is the axis for the per-axis ops
  // (group_id / local_id / …) and must be empty for the scalar ops
  // (group_size / wave_size). Returns null when the method is not a
  // launch-geo query; caller falls through.
  Value tryEmitLaunchGeo(StringRef method, Value group, Location loc,
                         std::optional<unsigned> resultIdx);

  // Call-site special-cased kwargs get picked out of the argument list
  // before operands are lowered to real `hc` ops, so the callee can see a
  // flat positional arg list plus a {name: attr} map.
  struct CallArgs {
    SmallVector<Value> positional;
    llvm::StringMap<Attribute> kwattrs;
    llvm::StringMap<Value> kwvalues;
  };
  FailureOr<CallArgs> collectCallArgs(hc_front::CallOp op);
};

//===----------------------------------------------------------------------===//
// Top-level callable dispatch.
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerCallable(Operation *frontOp) {
  MLIRContext *ctx = frontOp->getContext();
  Location loc = frontOp->getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(frontOp);

  // The `parameters = [...]` attribute is populated by the Python driver
  // for every callable and carries the ordered param list (plus optional
  // structural annotations). Downstream we rely on both the arity and the
  // name -> block-arg binding, so a missing attr is programmer error.
  auto params = frontOp->getAttrOfType<ArrayAttr>("parameters");
  if (!params) {
    return frontOp->emitOpError(
        "missing `parameters` attribute; the hc_front driver must stamp one");
  }

  // All three callable kinds share the same skeleton: build a fresh
  // entry block, bind params into a new Scope, hand (entry, fnType) to
  // the per-kind `build` lambda which creates the hc op and attaches
  // the block, then drive `lowerRegion` against the returned hc_front
  // body region.
  //
  // Ownership rule for `entry`: if the lambda returns success, it MUST
  // have attached the block to the hc op's body (the hc op then owns
  // it); on failure, it MUST either not have attached the block (we
  // delete here) or have erased the hc op (which takes the block with
  // it). The lambdas below follow the "attach first, fail with erase"
  // discipline to keep this obvious.
  auto runBody =
      [&](bool returnsValue,
          llvm::function_ref<FailureOr<Region *>(Block *, FunctionType)> build)
      -> LogicalResult {
    Block *entry = new Block();
    Scope scope;
    auto fnType =
        materializeParameters(params, *entry, scope, frontOp, returnsValue);
    if (failed(fnType)) {
      delete entry;
      return failure();
    }
    FailureOr<Region *> bodyRegion = build(entry, *fnType);
    if (failed(bodyRegion))
      return failure();
    scopes.push_back(std::make_unique<Scope>(std::move(scope)));
    OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(entry);
    LogicalResult r = lowerRegion(**bodyRegion);
    scopes.pop_back();
    return r;
  };

  if (auto kernel = dyn_cast<hc_front::KernelOp>(frontOp)) {
    return runBody(
        /*returnsValue=*/false,
        [&](Block *entry, FunctionType fnType) -> FailureOr<Region *> {
          auto hcKernel = HCKernelOp::create(
              builder, loc, StringAttr::get(ctx, kernel.getName()),
              TypeAttr::get(fnType), /*work_shape=*/ShapeAttr(),
              /*group_shape=*/ShapeAttr(),
              /*subgroup_size=*/IntegerAttr(), /*literals=*/ArrayAttr(),
              /*requirements=*/ConstraintSetAttr());
          hcKernel.getBody().push_back(entry);
          // Shape-like metadata travels as string arrays in hc_front;
          // convert to `#hc.shape<...>` for typed downstream consumers.
          if (auto ws = frontOp->getAttrOfType<ArrayAttr>("work_shape")) {
            auto shape = stringArrayToShape(ctx, frontOp, ws);
            if (failed(shape)) {
              hcKernel->erase();
              return failure();
            }
            hcKernel.setWorkShapeAttr(*shape);
          }
          if (auto gs = frontOp->getAttrOfType<ArrayAttr>("group_shape")) {
            auto shape = stringArrayToShape(ctx, frontOp, gs);
            if (failed(shape)) {
              hcKernel->erase();
              return failure();
            }
            hcKernel.setGroupShapeAttr(*shape);
          }
          if (auto sg = frontOp->getAttrOfType<IntegerAttr>("subgroup_size"))
            hcKernel.setSubgroupSizeAttr(sg);
          if (auto lits = frontOp->getAttrOfType<ArrayAttr>("literals"))
            hcKernel.setLiteralsAttr(lits);
          return &kernel.getBody();
        });
  }

  if (auto func = dyn_cast<hc_front::FuncOp>(frontOp)) {
    return runBody(
        /*returnsValue=*/true,
        [&](Block *entry, FunctionType fnType) -> FailureOr<Region *> {
          auto hcFunc = HCFuncOp::create(
              builder, loc, StringAttr::get(ctx, func.getName()),
              TypeAttr::get(fnType), /*requirements=*/ConstraintSetAttr(),
              /*effects=*/EffectClassAttr());
          hcFunc.getBody().push_back(entry);
          if (auto effAttr = frontOp->getAttrOfType<StringAttr>("effects")) {
            auto cls = parseEffectClass(effAttr.getValue());
            if (!cls) {
              hcFunc->emitOpError("unknown effects class '")
                  << effAttr.getValue() << "'";
              hcFunc->erase();
              return failure();
            }
            hcFunc.setEffectsAttr(EffectClassAttr::get(ctx, *cls));
          }
          // `scope` travels as a generic discardable attr on hc.func,
          // mirroring the existing use_scope_and_effects round-trip test.
          if (auto scopeAttr = frontOp->getAttrOfType<StringAttr>("scope"))
            hcFunc->setAttr("scope", ScopeAttr::get(ctx, scopeAttr.getValue()));
          return &func.getBody();
        });
  }

  if (auto intr = dyn_cast<hc_front::IntrinsicOp>(frontOp)) {
    auto scopeAttr = frontOp->getAttrOfType<StringAttr>("scope");
    if (!scopeAttr) {
      return frontOp->emitOpError(
          "hc_front.intrinsic must carry a `scope` string attribute");
    }
    return runBody(
        /*returnsValue=*/true,
        [&](Block *entry, FunctionType fnType) -> FailureOr<Region *> {
          auto hcIntr = HCIntrinsicOp::create(
              builder, loc, StringAttr::get(ctx, intr.getName()),
              TypeAttr::get(fnType), ScopeAttr::get(ctx, scopeAttr.getValue()),
              /*effects=*/EffectClassAttr(), /*const_kwargs=*/ArrayAttr());
          hcIntr.getBody().push_back(entry);
          if (auto effAttr = frontOp->getAttrOfType<StringAttr>("effects")) {
            auto cls = parseEffectClass(effAttr.getValue());
            if (!cls) {
              hcIntr->emitOpError("unknown effects class '")
                  << effAttr.getValue() << "'";
              hcIntr->erase();
              return failure();
            }
            hcIntr.setEffectsAttr(EffectClassAttr::get(ctx, *cls));
          }
          if (auto kw = frontOp->getAttrOfType<ArrayAttr>("const_kwargs"))
            hcIntr.setConstKwargsAttr(kw);
          return &intr.getBody();
        });
  }

  return frontOp->emitOpError("unexpected top-level hc_front op");
}

FailureOr<FunctionType>
Lowerer::materializeParameters(ArrayAttr params, Block &entry, Scope &scope,
                               Operation *sourceOp, bool returnsValue) {
  MLIRContext *ctx = sourceOp->getContext();
  SmallVector<Type> inputs;
  inputs.reserve(params.size());
  for (Attribute param : params) {
    auto dict = dyn_cast<DictionaryAttr>(param);
    if (!dict)
      return sourceOp->emitOpError(
                 "expected `parameters` entries to be DictAttr, got ")
             << param;
    auto name = dict.getAs<StringAttr>("name");
    if (!name)
      return sourceOp->emitOpError("`parameters` entry missing `name` key");
    // V0: every parameter is `!hc.undef`. The bead allows pinning typed
    // annotations, but we defer that to keep this bead's scope lean — a
    // later inference pass (or a follow-up bead) can walk the annotation
    // records and refine.
    inputs.push_back(undef);
    BlockArgument arg = entry.addArgument(undef, sourceOp->getLoc());
    scope.bind(name.getValue(), arg);
  }

  SmallVector<Type> results;
  if (returnsValue)
    results.push_back(undef);
  return FunctionType::get(ctx, inputs, results);
}

//===----------------------------------------------------------------------===//
// Per-region walk. Handles every `hc_front` op individually; unknown ops
// surface as a diagnostic rather than silent skip.
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerRegion(Region &src) {
  if (src.empty())
    return success();
  // The Python driver always emits single-block regions; multi-block
  // control flow would require CFG-aware scope merging we don't do yet.
  // Fail loud rather than silently skip the trailing blocks.
  if (!src.hasOneBlock()) {
    return src.getParentOp()->emitOpError(
        "hc_front regions must be single-block; multi-block lowering is not "
        "supported");
  }
  Block &block = src.front();
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (failed(lowerOp(&op)))
      return failure();
  }
  return success();
}

LogicalResult Lowerer::lowerOp(Operation *op) {
  // `target_name`, `target_tuple`, `target_subscript` don't emit `hc` ops
  // in their own right — they only matter as the LHS of an `hc_front.assign`
  // where the assign pattern walks their defining ops. Recording them here
  // as null in the value map keeps operand lookups from tripping on the
  // missing mapping.
  if (isa<hc_front::TargetNameOp, hc_front::TargetTupleOp,
          hc_front::TargetSubscriptOp>(op)) {
    valueMap[op->getResult(0)] = Value();
    return success();
  }

  // Name / attr ops are the classification surface of the `ref` metadata.
  // Some classifications (callee/intrinsic/inline/builtin/module/numpy_*)
  // are consumed at the call site; others (param/iv/local/constant/symbol)
  // immediately resolve to a concrete `hc` value. `lowerName` returns
  // success+null for the former (the call-site lookup inspects the
  // defining op directly) and failure() for real errors.
  if (auto name = dyn_cast<hc_front::NameOp>(op)) {
    FailureOr<Value> v = lowerName(name);
    if (failed(v))
      return failure();
    valueMap[name.getResult()] = *v;
    return success();
  }
  if (auto attr = dyn_cast<hc_front::AttrOp>(op)) {
    valueMap[attr.getResult()] = lowerAttr(attr);
    return success();
  }

  if (auto c = dyn_cast<hc_front::ConstantOp>(op)) {
    valueMap[c.getResult()] = lowerConstant(c);
    return success();
  }
  if (auto b = dyn_cast<hc_front::BinOp>(op)) {
    valueMap[b.getResult()] = lowerBinop(b);
    return valueMap[b.getResult()] ? success() : failure();
  }
  if (auto s = dyn_cast<hc_front::SliceOp>(op)) {
    Value v = lowerSlice(s);
    valueMap[s.getResult()] = v;
    return v ? success() : failure();
  }
  if (auto r = dyn_cast<hc_front::ReturnOp>(op))
    return lowerReturn(r);
  if (auto a = dyn_cast<hc_front::AssignOp>(op))
    return lowerAssign(a);
  if (auto f = dyn_cast<hc_front::ForOp>(op))
    return lowerFor(f);
  if (auto w = dyn_cast<hc_front::WorkitemRegionOp>(op))
    return lowerWorkitemRegion(w);
  if (auto sg = dyn_cast<hc_front::SubgroupRegionOp>(op))
    return lowerSubgroupRegion(sg);
  if (auto c = dyn_cast<hc_front::CallOp>(op)) {
    FailureOr<Value> v = lowerCall(c);
    if (failed(v))
      return failure();
    // `v` may be null for calls that the parent op consumes directly
    // (e.g. `range(...)` inside `hc_front.for`). That's still a success.
    valueMap[c.getResult()] = *v;
    return success();
  }
  if (auto s = dyn_cast<hc_front::SubscriptOp>(op)) {
    valueMap[s.getResult()] = lowerSubscript(s);
    return valueMap[s.getResult()] ? success() : failure();
  }
  if (auto t = dyn_cast<hc_front::TupleOp>(op)) {
    SmallVector<Value> elts = lowerValueOperands(t.getElements());
    tupleElts[t.getResult()] = elts;
    valueMap[t.getResult()] = Value();
    return success();
  }
  if (auto k = dyn_cast<hc_front::KeywordOp>(op)) {
    Value v = lowerValueOperand(k.getValue());
    keywordInfo[k.getResult()] = {k.getName(), k.getValue(), v};
    valueMap[k.getResult()] = Value();
    return success();
  }

  return op->emitOpError("unsupported hc_front op");
}

Value Lowerer::lowerValueOperand(Value v) {
  auto it = valueMap.find(v);
  assert(it != valueMap.end() && "operand was not yet lowered; walk order bug");
  return it->second;
}

SmallVector<Value> Lowerer::lowerValueOperands(ValueRange vs) {
  SmallVector<Value> out;
  out.reserve(vs.size());
  for (Value v : vs)
    out.push_back(lowerValueOperand(v));
  return out;
}

SmallVector<Value> Lowerer::expandTupleOperand(Value v) {
  // If the operand is an `hc_front.tuple` the operand list lives in
  // `tupleElts`; otherwise treat it as a singleton. This lets variadic
  // consumers (subscript indices, call args) stay uniform.
  auto it = tupleElts.find(v);
  if (it != tupleElts.end())
    return it->second;
  Value lowered = lowerValueOperand(v);
  return lowered ? SmallVector<Value>{lowered} : SmallVector<Value>{};
}

//===----------------------------------------------------------------------===//
// Name / attr. The classification on `ref` drives everything.
//===----------------------------------------------------------------------===//

FailureOr<Value> Lowerer::lowerName(hc_front::NameOp op) {
  RefInfo ref = RefInfo::of(op);
  StringRef kind = ref.kind;
  StringRef ident = op.getName();
  if (kind == "param" || kind == "local" || kind == "iv") {
    Value v = currentScope().lookup(ident);
    if (!v) {
      op.emitOpError("hc_front.name '")
          << ident << "' (kind=" << kind
          << ") not bound in the current scope; missing assign or bad ref";
      return failure();
    }
    return v;
  }
  if (kind == "constant") {
    Value v = emitConstantFromRef(op.getLoc(), ref, op);
    if (!v)
      return failure();
    return v;
  }
  if (kind == "symbol") {
    // `#hc.expr<"Ident">` is the pinned form for a bare symbol; the
    // resulting `!hc.idx<"Ident">` type uniquely identifies the symbol.
    auto &store =
        op->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
    std::string diag;
    FailureOr<sym::ExprHandle> handle = sym::parseExpr(store, ident, &diag);
    if (failed(handle)) {
      op.emitOpError("symbol '") << ident << "' failed to parse: " << diag;
      return failure();
    }
    auto expr = ExprAttr::get(op.getContext(), *handle);
    Type idxTy = IdxType::get(op.getContext(), expr);
    return HCSymbolOp::create(builder, op.getLoc(), idxTy).getResult();
  }
  // callee / intrinsic / inline / builtin / module / numpy_* — the call
  // dispatcher and subscript pattern read these off the original op.
  // Success with a null Value is the intentional "no SSA result" sentinel.
  return Value();
}

Value Lowerer::lowerAttr(hc_front::AttrOp op) {
  // Attribute chains like `np.float32` or `group.load` are either folded
  // into a parent call (dsl_method, numpy_dtype_type), or don't produce a
  // value at all. Leave nothing in the value map — consumers inspect the
  // original op.
  return Value();
}

Value Lowerer::emitConstantFromRef(Location loc, const RefInfo &ref,
                                   Operation *sourceOp) {
  MLIRContext *ctx = sourceOp->getContext();
  StringRef pyKind = ref.string("python_kind");
  StringRef raw = ref.string("value");
  Attribute payload;
  if (pyKind == "int") {
    int64_t v = 0;
    if (raw.getAsInteger(10, v)) {
      sourceOp->emitOpError("constant value '")
          << raw << "' not parseable as int";
      return nullptr;
    }
    payload = IntegerAttr::get(IntegerType::get(ctx, 64), v);
  } else if (pyKind == "float") {
    APFloat f(APFloat::IEEEdouble());
    auto status = f.convertFromString(raw, APFloat::rmNearestTiesToEven);
    if (!status) {
      llvm::consumeError(status.takeError());
      sourceOp->emitOpError("constant value '")
          << raw << "' not parseable as float";
      return nullptr;
    }
    payload = FloatAttr::get(Float64Type::get(ctx), f);
  } else if (pyKind == "bool") {
    payload = BoolAttr::get(ctx, raw == "True");
  } else if (pyKind == "str") {
    // The driver wraps strings in Python repr quotes; trim the outer
    // single quotes so the attribute carries the raw text.
    StringRef s = raw;
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'')
      s = s.drop_front().drop_back();
    payload = StringAttr::get(ctx, s);
  } else {
    sourceOp->emitOpError("unsupported constant python_kind '")
        << pyKind << "'";
    return nullptr;
  }
  return HCConstOp::create(builder, loc, undef, payload);
}

//===----------------------------------------------------------------------===//
// Simple value producers.
//===----------------------------------------------------------------------===//

Value Lowerer::lowerConstant(hc_front::ConstantOp op) {
  return HCConstOp::create(builder, op.getLoc(), undef, op.getValue());
}

Value Lowerer::lowerBinop(hc_front::BinOp op) {
  Value lhs = lowerValueOperand(op.getLhs());
  Value rhs = lowerValueOperand(op.getRhs());
  if (!lhs || !rhs) {
    op.emitOpError("binop has unresolved operand");
    return nullptr;
  }
  return emitBinop(builder, op.getLoc(), op.getKind(), lhs, rhs, undef, op);
}

Value Lowerer::lowerSlice(hc_front::SliceOp op) {
  // `has_*` flags tell us which optional parts were syntactically present;
  // the operand list packs only the present parts, in (lower, upper, step)
  // order. `hc.slice_expr` mirrors the tri-state via `Optional<>` operands.
  // Missing attributes on hand-written IR default to false rather than a
  // null-dereference crash.
  auto boolAttr = [&](StringRef key) -> bool {
    auto a = op->getAttrOfType<BoolAttr>(key);
    return a ? a.getValue() : false;
  };
  bool hasLower = boolAttr("has_lower");
  bool hasUpper = boolAttr("has_upper");
  bool hasStep = boolAttr("has_step");

  SmallVector<Value> parts = lowerValueOperands(op.getParts());
  size_t expected = unsigned(hasLower) + unsigned(hasUpper) + unsigned(hasStep);
  if (parts.size() != expected) {
    op.emitOpError("slice has_* flags (")
        << expected << ") disagree with part count (" << parts.size() << ")";
    return nullptr;
  }

  Value lo = nullptr, hi = nullptr, st = nullptr;
  size_t idx = 0;
  if (hasLower)
    lo = parts[idx++];
  if (hasUpper)
    hi = parts[idx++];
  if (hasStep)
    st = parts[idx++];
  (void)idx;

  return HCSliceExprOp::create(builder, op.getLoc(), undef, lo, hi, st);
}

//===----------------------------------------------------------------------===//
// Control flow / binding.
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerReturn(hc_front::ReturnOp op) {
  SmallVector<Value> values = lowerValueOperands(op.getValues());
  HCReturnOp::create(builder, op.getLoc(), values);
  return success();
}

LogicalResult Lowerer::lowerAssign(hc_front::AssignOp op) {
  Value rhs = op.getValue();
  Operation *target = op.getTarget().getDefiningOp();

  if (auto tn = dyn_cast_if_present<hc_front::TargetNameOp>(target)) {
    Value value = lowerValueOperand(rhs);
    currentScope().bind(tn.getName(), value);
    return success();
  }

  if (auto tt = dyn_cast_if_present<hc_front::TargetTupleOp>(target)) {
    // Multi-assign: `a, b = <rhs>`. Two shapes we accept:
    //   (1) RHS is an `hc_front.tuple` — destructure from `tupleElts`.
    //       `lowerValueOperand` on a tuple returns null (tuples have no
    //       standalone `hc` counterpart), so we must special-case before
    //       looking at the lowered value or we'd bind every target to null.
    //   (2) RHS lowers to a single op whose result count matches the
    //       target arity (a call that happens to produce N results).
    size_t arity = tt.getElements().size();
    SmallVector<Value> sources;
    auto tupleIt = tupleElts.find(rhs);
    if (tupleIt != tupleElts.end()) {
      sources.assign(tupleIt->second.begin(), tupleIt->second.end());
    } else {
      Value value = lowerValueOperand(rhs);
      Operation *rhsOp = value ? value.getDefiningOp() : nullptr;
      if (rhsOp && rhsOp->getNumResults() == arity) {
        sources.assign(rhsOp->getResults().begin(), rhsOp->getResults().end());
      } else {
        return op.emitOpError(
            "tuple-unpack rhs must be an hc_front.tuple or an arity-matching "
            "multi-result op");
      }
    }
    if (sources.size() != arity)
      return op.emitOpError("tuple-unpack arity mismatch: rhs has ")
             << sources.size() << ", target has " << arity;
    for (auto [elem, src] : llvm::zip(tt.getElements(), sources)) {
      auto tn =
          dyn_cast_if_present<hc_front::TargetNameOp>(elem.getDefiningOp());
      if (!tn)
        return op.emitOpError("nested target kinds are not yet supported");
      if (!src)
        return op.emitOpError(
            "tuple-unpack source element did not lower to an hc value");
      currentScope().bind(tn.getName(), src);
    }
    return success();
  }

  if (auto ts = dyn_cast_if_present<hc_front::TargetSubscriptOp>(target)) {
    Value value = lowerValueOperand(rhs);
    if (!value)
      return op.emitOpError("store source did not lower to an hc value");
    Value base = lowerValueOperand(ts.getBase());
    if (!base)
      return op.emitOpError("target_subscript base is unresolved");
    SmallVector<Value> indices;
    for (Value idx : ts.getIndices()) {
      SmallVector<Value> expanded = expandTupleOperand(idx);
      indices.append(expanded.begin(), expanded.end());
    }
    HCStoreOp::create(builder, op.getLoc(), base, indices, value);
    return success();
  }

  return op.emitOpError("unsupported assign target");
}

LogicalResult Lowerer::lowerFor(hc_front::ForOp op) {
  // The iter region must be a `hc_front.call` to the `range` builtin. We
  // lower that region *eagerly* (before the main loop body) into the
  // enclosing block, then harvest the lo/hi/step operands. This keeps the
  // induction variable one consistent producer rather than a mix of
  // pre-loop and loop-body ops.
  Region &iter = op.getIter();
  if (iter.empty() || iter.front().empty())
    return op.emitOpError("for-iter region is empty");

  hc_front::CallOp iterCall;
  Operation *lastOp = nullptr;
  for (Operation &child : llvm::make_early_inc_range(iter.front())) {
    if (failed(lowerOp(&child)))
      return failure();
    lastOp = &child;
  }
  iterCall = dyn_cast_if_present<hc_front::CallOp>(lastOp);
  if (!iterCall)
    return op.emitOpError("for-iter must end in a call op");

  auto calleeName = dyn_cast_if_present<hc_front::NameOp>(
      iterCall.getCallee().getDefiningOp());
  RefInfo calleeRef = RefInfo::of(calleeName);
  if (!calleeName || calleeRef.kind != "builtin" ||
      calleeRef.string("builtin") != "range") {
    return op.emitOpError("for-iter must be `range(...)`");
  }

  // Python-style `range(stop)` / `range(start, stop)` / `range(start, stop,
  // step)`: pad the missing parts with the canonical defaults so the
  // `hc.for_range` op always sees three operands.
  SmallVector<Value> rangeArgs = lowerValueOperands(iterCall.getArguments());
  Value lo, hi, step;
  if (rangeArgs.size() == 1) {
    lo = HCConstOp::create(
        builder, op.getLoc(), undef,
        IntegerAttr::get(IntegerType::get(op.getContext(), 64), 0));
    hi = rangeArgs[0];
    step = HCConstOp::create(
        builder, op.getLoc(), undef,
        IntegerAttr::get(IntegerType::get(op.getContext(), 64), 1));
  } else if (rangeArgs.size() == 2) {
    lo = rangeArgs[0];
    hi = rangeArgs[1];
    step = HCConstOp::create(
        builder, op.getLoc(), undef,
        IntegerAttr::get(IntegerType::get(op.getContext(), 64), 1));
  } else if (rangeArgs.size() == 3) {
    lo = rangeArgs[0];
    hi = rangeArgs[1];
    step = rangeArgs[2];
  } else {
    return op.emitOpError("range(...) must have 1, 2, or 3 args");
  }

  // Pull the IV name out of the target region.
  Region &tgt = op.getTarget();
  if (tgt.empty() || tgt.front().empty())
    return op.emitOpError("for-target region is empty");
  auto ivTarget = dyn_cast<hc_front::TargetNameOp>(&tgt.front().front());
  if (!ivTarget)
    return op.emitOpError("for-target must be a single target_name");

  // Build the `hc.for_range` with no iter_args (loop-carried analysis is
  // 5-3jb). The entry block owns the IV block argument that we bind into
  // the child scope so name lookups inside the body resolve.
  auto forOp = HCForRangeOp::create(builder, op.getLoc(),
                                    /*resultTypes=*/TypeRange{}, lo, hi, step,
                                    /*iter_inits=*/ValueRange{});
  Block *body = new Block();
  BlockArgument iv = body->addArgument(undef, op.getLoc());
  forOp.getBody().push_back(body);

  Scope childScope;
  childScope.parent = scopes.back().get();
  childScope.bind(ivTarget.getName(), iv);
  scopes.push_back(std::make_unique<Scope>(std::move(childScope)));

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    LogicalResult r = lowerRegion(op.getBody());
    if (succeeded(r))
      HCYieldOp::create(builder, op.getLoc());
    scopes.pop_back();
    if (failed(r))
      return failure();
  }
  return success();
}

template <typename HCRegionOpT, typename FrontRegionOpT>
LogicalResult Lowerer::lowerCapturingRegion(FrontRegionOpT op) {
  // `hc.{workitem,subgroup}_region` match `hc_front` 1:1 (captures +
  // body). Nested-def folding is 5-3jb; if the front op declares
  // parameters we still bind them as block args so the body resolves,
  // but the `hc` op carries only a captures list (no formal params).
  auto newOp = HCRegionOpT::create(builder, op.getLoc(), op.getCapturesAttr());
  Block *body = new Block();
  Scope childScope;
  childScope.parent = scopes.back().get();
  if (auto params = op->template getAttrOfType<ArrayAttr>("parameters")) {
    for (Attribute p : params) {
      auto dict = dyn_cast<DictionaryAttr>(p);
      if (!dict)
        return op.emitOpError("invalid parameters entry");
      auto name = dict.template getAs<StringAttr>("name");
      if (!name)
        return op.emitOpError("parameters entry missing `name`");
      BlockArgument arg = body->addArgument(undef, op.getLoc());
      childScope.bind(name.getValue(), arg);
    }
  }
  newOp.getBody().push_back(body);

  scopes.push_back(std::make_unique<Scope>(std::move(childScope)));
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  LogicalResult r = lowerRegion(op.getBody());
  scopes.pop_back();
  return r;
}

LogicalResult Lowerer::lowerWorkitemRegion(hc_front::WorkitemRegionOp op) {
  return lowerCapturingRegion<HCWorkitemRegionOp>(op);
}

LogicalResult Lowerer::lowerSubgroupRegion(hc_front::SubgroupRegionOp op) {
  return lowerCapturingRegion<HCSubgroupRegionOp>(op);
}

//===----------------------------------------------------------------------===//
// Call / subscript.
//===----------------------------------------------------------------------===//

FailureOr<Lowerer::CallArgs> Lowerer::collectCallArgs(hc_front::CallOp op) {
  CallArgs args;
  for (Value arg : op.getArguments()) {
    auto kwIt = keywordInfo.find(arg);
    if (kwIt != keywordInfo.end()) {
      // Shape-like kwargs (`shape = (M, N)`) are folded to a shape attr
      // at the call site; other kwargs flow through as SSA operands (for
      // intrinsic const_kwargs) or produce errors later if unhandled.
      StringRef kwName = kwIt->second.name;
      Value kwOrig = kwIt->second.origValue;
      Value kwVal = kwIt->second.loweredValue;
      if (kwName == "shape") {
        // The tuple-of-constants -> `#hc.shape<...>` fold. The keyword's
        // *original* hc_front value is the tuple op; look up in
        // `tupleElts` by that key (the lowered value is null for tuples).
        // If the elements aren't all integer/string hc.const producers,
        // fall back to storing the (likely null) value so a later
        // lowering can still see it.
        auto tupleIt = tupleElts.find(kwOrig);
        if (tupleIt != tupleElts.end()) {
          SmallVector<Attribute> dims;
          dims.reserve(tupleIt->second.size());
          auto &store =
              op->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
          // If every element is an `hc.const` carrying an integer, we can
          // materialize a shape attribute. String-valued constants are
          // also accepted (they already name a symbol).
          bool ok = true;
          for (Value e : tupleIt->second) {
            Attribute dim;
            if (auto constOp = e.getDefiningOp<HCConstOp>()) {
              Attribute v = constOp.getValue();
              std::string text;
              if (auto i = dyn_cast<IntegerAttr>(v)) {
                text = std::to_string(i.getInt());
              } else if (auto s = dyn_cast<StringAttr>(v)) {
                text = s.getValue().str();
              } else {
                ok = false;
                break;
              }
              std::string diag;
              FailureOr<sym::ExprHandle> handle =
                  sym::parseExpr(store, text, &diag);
              if (failed(handle)) {
                ok = false;
                break;
              }
              dim = ExprAttr::get(op.getContext(), *handle);
            } else {
              ok = false;
              break;
            }
            dims.push_back(dim);
          }
          if (ok) {
            args.kwattrs["shape"] = ShapeAttr::get(op.getContext(), dims);
            continue;
          }
        }
      }
      args.kwvalues[kwName] = kwVal;
      continue;
    }
    Value lowered = lowerValueOperand(arg);
    if (!lowered) {
      op.emitOpError("call argument did not lower to an hc value");
      return failure();
    }
    args.positional.push_back(lowered);
  }
  return args;
}

FailureOr<Value> Lowerer::lowerCall(hc_front::CallOp op) {
  Operation *calleeDef = op.getCallee().getDefiningOp();
  // DSL-method call: `%m = hc_front.attr %base, "method"` + `hc_front.call %m`.
  if (auto attrOp = dyn_cast_if_present<hc_front::AttrOp>(calleeDef))
    return lowerDslMethodCall(op, attrOp);

  auto nameOp = dyn_cast_if_present<hc_front::NameOp>(calleeDef);
  if (!nameOp) {
    op.emitOpError("call with non-name, non-attr callee not supported");
    return failure();
  }
  RefInfo ref = RefInfo::of(nameOp);
  StringRef kind = ref.kind;

  FailureOr<CallArgs> argsOr = collectCallArgs(op);
  if (failed(argsOr))
    return failure();
  CallArgs &args = *argsOr;

  if (kind == "callee" || kind == "intrinsic") {
    StringRef callee = ref.string("callee");
    if (callee.empty()) {
      op.emitOpError("`ref.callee` missing for ") << kind << " name ref";
      return failure();
    }
    // Strip the leading "@" the Python driver stamps so we can hand a
    // bare symbol name to `FlatSymbolRefAttr::get` (which re-adds it).
    if (callee.starts_with("@"))
      callee = callee.drop_front();
    auto symRef = FlatSymbolRefAttr::get(op.getContext(), callee);

    // `hc` calls carry a single result (pre-inference = `!hc.undef`); if a
    // downstream multi-assign wants more, the assign pattern detects it
    // from the RHS op and distributes. `hc.call_intrinsic` additionally
    // promotes `const_kwargs` into attributes on the op.
    if (kind == "callee") {
      auto call = HCCallOp::create(builder, op.getLoc(), TypeRange{undef},
                                   symRef, args.positional);
      return call.getResult(0);
    }
    auto call = HCCallIntrinsicOp::create(
        builder, op.getLoc(), TypeRange{undef}, symRef, args.positional);
    if (auto constKwargs = ref.getAs<ArrayAttr>("const_kwargs")) {
      // `const_kwargs` is the intrinsic's declared list of kwargs that must
      // be constant-folded into attributes on the call site. Each entry we
      // find in the call's kwargs lands as either (a) the already-folded
      // attribute (shape fold from collectCallArgs) or (b) the payload of
      // the `hc.const` producer.
      for (Attribute kw : constKwargs) {
        auto kwStr = dyn_cast<StringAttr>(kw);
        if (!kwStr)
          continue;
        StringRef kwName = kwStr.getValue();
        auto attrIt = args.kwattrs.find(kwName);
        if (attrIt != args.kwattrs.end()) {
          call->setAttr(kwName, attrIt->second);
          continue;
        }
        auto valIt = args.kwvalues.find(kwName);
        if (valIt == args.kwvalues.end())
          continue;
        // Const-kwargs must be constant, and hc.const is the canonical
        // producer; fish the payload attribute off the const op so the
        // kwarg lands as an attribute, not an SSA operand.
        if (auto constOp = valIt->second.getDefiningOp<HCConstOp>())
          call->setAttr(kwName, constOp.getValue());
      }
    }
    return call.getResult(0);
  }
  if (kind == "builtin") {
    // Consumed-by-parent builtins (right now: `range`, always folded by the
    // for-loop lowering). No stand-alone hc op. Returning a null Value is
    // safe because the only consumers walk the original hc_front op.
    return Value();
  }
  if (kind == "inline") {
    // Full inlining of non-decorated helpers needs a separate pass (see
    // bead note). Emit a conservative `hc.call @<qualified>` placeholder
    // and let downstream either inline it or flag the missing symbol.
    StringRef qn = ref.string("qualified_name");
    if (qn.empty()) {
      op.emitOpError("inline name ref missing qualified_name");
      return failure();
    }
    // Collapse dotted paths to the short name so the symbol ref is a
    // legal flat symbol; downstream inlining can consult the original
    // `ref` payload if it needs the fully-qualified lookup.
    StringRef shortName = qn;
    if (auto pos = qn.rfind('.'); pos != StringRef::npos)
      shortName = qn.drop_front(pos + 1);
    auto symRef = FlatSymbolRefAttr::get(op.getContext(), shortName);
    auto call = HCCallOp::create(builder, op.getLoc(), TypeRange{undef}, symRef,
                                 args.positional);
    return {call.getResult(0)};
  }
  op.emitOpError("unsupported callee ref.kind '") << kind << "'";
  return failure();
}

FailureOr<Value> Lowerer::lowerDslMethodCall(hc_front::CallOp call,
                                             hc_front::AttrOp attr) {
  RefInfo ref = RefInfo::of(attr);
  StringRef method = ref.string("method");

  FailureOr<CallArgs> argsOr = collectCallArgs(call);
  if (failed(argsOr))
    return failure();
  CallArgs &args = *argsOr;

  Value base = lowerValueOperand(attr.getBase());

  // `x.vec()`, `x.with_inactive(value=...)`, `x.astype(target)` are the
  // mechanical cases this bead handles. Anything requiring group/context
  // plumbing (`wi.local_id()`, `group.load(...)`) is deferred — the
  // acceptance synthetic test sticks to the mechanical subset.
  // All unary-base DSL methods share the "base did not lower" guard; a
  // classification gap must not let us ship an hc op built on a null
  // operand — the later verifier error would be harder to attribute.
  auto requireBase = [&](StringRef m) -> LogicalResult {
    if (!base) {
      call.emitOpError(m) << ": base did not lower";
      return failure();
    }
    return success();
  };

  if (method == "vec") {
    if (failed(requireBase(method)))
      return failure();
    return {HCVecOp::create(builder, call.getLoc(), undef, base).getResult()};
  }
  if (method == "with_inactive") {
    if (failed(requireBase(method)))
      return failure();
    auto valIt = args.kwvalues.find("value");
    if (valIt == args.kwvalues.end()) {
      call.emitOpError("with_inactive missing `value=` kwarg");
      return failure();
    }
    Attribute inactive;
    if (auto constOp = valIt->second.getDefiningOp<HCConstOp>())
      inactive = constOp.getValue();
    if (!inactive) {
      call.emitOpError("with_inactive value must be an hc.const");
      return failure();
    }
    return {
        HCWithInactiveOp::create(builder, call.getLoc(), undef, base, inactive)
            .getResult()};
  }
  if (method == "astype") {
    if (failed(requireBase(method)))
      return failure();
    if (args.positional.empty()) {
      call.emitOpError("astype missing target type");
      return failure();
    }
    // The frontend passes the target dtype as the first positional arg
    // (a numpy_dtype_type name -> attr). We accept either an hc.const
    // carrying a TypeAttr, or a plain TypeAttr stashed on the source name
    // op. Both collapse to a TypeAttr here.
    Value target = args.positional.front();
    TypeAttr targetAttr;
    if (auto constOp = target.getDefiningOp<HCConstOp>()) {
      if (auto t = dyn_cast<TypeAttr>(constOp.getValue()))
        targetAttr = t;
    }
    if (!targetAttr) {
      call.emitOpError("astype target must resolve to a TypeAttr");
      return failure();
    }
    return {HCAsTypeOp::create(builder, call.getLoc(), undef, base, targetAttr)
                .getResult()};
  }
  // Buffer/tensor load, vload, store: the frontend passes a (possibly
  // pre-sliced) handle as the first positional, remaining positionals go
  // to the indices list, and an optional `shape=` kwarg lands as the
  // attribute. We rely on `collectCallArgs`' shape-kwarg fold so no manual
  // tuple walking is needed here.
  if (method == "load" || method == "vload") {
    if (args.positional.empty()) {
      call.emitOpError("`") << method << "` expects a buffer/tensor argument";
      return failure();
    }
    Value src = args.positional.front();
    SmallVector<Value> indices(args.positional.begin() + 1,
                               args.positional.end());
    auto shapeAttr = dyn_cast_or_null<ShapeAttr>(args.kwattrs.lookup("shape"));
    Operation *op = method == "load"
                        ? HCLoadOp::create(builder, call.getLoc(), undef, src,
                                           indices, shapeAttr)
                              .getOperation()
                        : HCVLoadOp::create(builder, call.getLoc(), undef, src,
                                            indices, shapeAttr)
                              .getOperation();
    return {op->getResult(0)};
  }
  if (method == "store") {
    if (args.positional.size() < 2) {
      call.emitOpError("`store` expects at least (dest, source)");
      return failure();
    }
    Value dest = args.positional.front();
    Value source = args.positional.back();
    SmallVector<Value> indices(args.positional.begin() + 1,
                               args.positional.end() - 1);
    HCStoreOp::create(builder, call.getLoc(), dest, indices, source);
    // `hc.store` is a no-result op; success+null signals "consumed, no
    // SSA output" to `lowerCall`, distinct from the failure path below.
    return Value();
  }
  if (method == "vzeros") {
    auto shapeAttr = dyn_cast_or_null<ShapeAttr>(args.kwattrs.lookup("shape"));
    if (!shapeAttr) {
      call.emitOpError("`vzeros` missing or invalid `shape=` kwarg");
      return failure();
    }
    return {HCVZerosOp::create(builder, call.getLoc(), undef, shapeAttr)
                .getResult()};
  }

  // `group.{launch_geo}()` with no args is the call form (`wi.local_id()`).
  // Emit the multi-result launch-geo op; the caller's enclosing subscript
  // drills into a specific axis. The property-style `group.launch_geo[N]`
  // is handled in `lowerSubscript` — both go through `tryEmitLaunchGeo`
  // so the supported method set stays in one place.
  if (call.getArguments().empty() && args.kwvalues.empty() &&
      args.kwattrs.empty()) {
    if (base) {
      // We do not know the launch rank at this site, so emit a single-
      // result op for now and rely on a later canonicalizer + rank
      // inference to widen it as needed.
      if (Value v = tryEmitLaunchGeo(method, base, call.getLoc(),
                                     /*resultIdx=*/std::nullopt))
        return {v};
    }
  }
  call.emitOpError("unsupported dsl_method '") << method << "'";
  return failure();
}

Value Lowerer::tryEmitLaunchGeo(StringRef method, Value group, Location loc,
                                std::optional<unsigned> resultIdx) {
  auto emitMulti = [&](auto tag) -> Value {
    using OpT = decltype(tag);
    unsigned n = resultIdx.value_or(0) + 1;
    SmallVector<Type> resTypes(n, undef);
    auto op = OpT::create(builder, loc, resTypes, group);
    return op.getResult(resultIdx.value_or(0));
  };
  auto emitScalar = [&](auto tag) -> Value {
    using OpT = decltype(tag);
    auto op = OpT::create(builder, loc, undef, group);
    return op.getResult(0);
  };
  if (method == "group_id")
    return emitMulti(HCGroupIdOp{});
  if (method == "local_id")
    return emitMulti(HCLocalIdOp{});
  if (method == "subgroup_id")
    return emitMulti(HCSubgroupIdOp{});
  if (method == "group_shape")
    return emitMulti(HCGroupShapeOp{});
  if (method == "work_offset")
    return emitMulti(HCWorkOffsetOp{});
  if (method == "work_shape")
    return emitMulti(HCWorkShapeOp{});
  if (method == "group_size")
    return emitScalar(HCGroupSizeOp{});
  if (method == "wave_size")
    return emitScalar(HCWaveSizeOp{});
  return {};
}

Value Lowerer::lowerSubscript(hc_front::SubscriptOp op) {
  // DSL-method `[]` patterns fold into dedicated hc ops when the base is an
  // `hc_front.attr` and the index is a single integer constant:
  //   x.shape[N]      -> hc.buffer_dim
  //   group.group_id[N] / local_id[N] / ... -> hc.<launch_geo>
  // Property-style (no `()`) access to launch geometry lands here, in
  // contrast to the call-style (`wi.local_id()[N]`) which routes through
  // `lowerDslMethodCall` + a subsequent subscript that falls to the
  // generic `hc.buffer_view` branch below.
  if (auto attr =
          dyn_cast_if_present<hc_front::AttrOp>(op.getBase().getDefiningOp())) {
    RefInfo ref = RefInfo::of(attr);
    StringRef method = ref.string("method");
    if (!method.empty() && op.getIndices().size() == 1) {
      Value idxVal = lowerValueOperand(op.getIndices().front());
      Value baseVal = lowerValueOperand(attr.getBase());
      auto constOp = idxVal ? idxVal.getDefiningOp<HCConstOp>() : nullptr;
      auto ax =
          constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr{};
      if (baseVal && ax) {
        int64_t axVal = ax.getInt();
        // Launch-geo / buffer_dim axes are small non-negative integers;
        // anything else is either user error or a bad `ref` payload.
        // Guard before the unsigned cast so a negative never underflows
        // into a huge variadic-op rank below.
        if (axVal < 0 || axVal >= kMaxLaunchAxis) {
          op.emitOpError("axis index ")
              << axVal << " out of range [0, " << kMaxLaunchAxis << ")";
          return nullptr;
        }
        unsigned axis = static_cast<unsigned>(axVal);
        if (method == "shape") {
          return HCBufferDimOp::create(
              builder, op.getLoc(), undef, baseVal,
              IntegerAttr::get(IntegerType::get(op.getContext(), 64), axVal));
        }
        if (Value v = tryEmitLaunchGeo(method, baseVal, op.getLoc(), axis))
          return v;
      }
    }
  }

  Value base = lowerValueOperand(op.getBase());
  if (!base) {
    op.emitOpError("subscript base did not lower");
    return nullptr;
  }
  SmallVector<Value> indices;
  for (Value idx : op.getIndices()) {
    SmallVector<Value> expanded = expandTupleOperand(idx);
    indices.append(expanded.begin(), expanded.end());
  }
  // `hc.buffer_view` accepts `!hc.undef` as the buffer handle via
  // `HC_BufferValueType`, so pre-inference subscripts pass verify.
  return HCBufferViewOp::create(builder, op.getLoc(), undef, base, indices);
}

//===----------------------------------------------------------------------===//
// Pass scaffolding. The pass runs on the enclosing `builtin.module`, walks
// every `hc_front` top-level callable, and erases it once the parallel
// `hc` callable has been emitted.
//===----------------------------------------------------------------------===//

struct ConvertHCFrontToHCPass
    : public hc_front::impl::ConvertHCFrontToHCBase<ConvertHCFrontToHCPass> {
  using ConvertHCFrontToHCBase::ConvertHCFrontToHCBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    UndefType undef = UndefType::get(ctx);

    SmallVector<Operation *> frontOps;
    for (Operation &op : *module.getBody()) {
      if (isa<hc_front::KernelOp, hc_front::FuncOp, hc_front::IntrinsicOp>(op))
        frontOps.push_back(&op);
    }

    OpBuilder builder(ctx);
    for (Operation *op : frontOps) {
      Lowerer lowerer(builder, undef);
      if (failed(lowerer.lowerCallable(op))) {
        signalPassFailure();
        return;
      }
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> hc_front::createConvertHCFrontToHCPass() {
  return std::make_unique<ConvertHCFrontToHCPass>();
}
