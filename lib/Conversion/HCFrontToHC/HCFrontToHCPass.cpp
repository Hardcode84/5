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
// Python-level name bindings are emitted as `hc.assign` / `hc.name_load`
// placeholder ops; `-hc-promote-names` then folds them into SSA. See
// `ConvertHCFrontToHC` in `include/hc/Conversion/HCFrontToHC/Passes.td`
// for the canonical pipeline contract — this banner doesn't repeat it.
//
// What this pass handles today:
//  * the structural rewrites for every `hc_front` op the WMMA kernel touches
//    in v0 (kernel/func/intrinsic, workitem/subgroup regions, for-range over
//    a `range(...)` iter, constant, binop, name dispatched on `ref`,
//    target_* + assign, slice, tuple/keyword glue, subscript, call dispatched
//    on the callee's `ref`, plus a small `dsl_method` subset that's mechanical
//    enough to fit — `a.shape[N]`, `x.vec()`, `x.with_inactive(value=...)`,
//    `x.astype(...)`);
//  * DSL method dispatch reads `hc_front.attr`'s `$name` (the op-level
//    spelling the frontend always stamps) rather than `ref.method`,
//    which the resolver can only fill in when the attr's base was
//    classifiable — chained attrs on subscript/call results arrive with
//    no `ref.method` but still dispatch on the attr's name;
//  * `numpy_dtype_type` attrs (`np.float32`, `np.float16`, ...) lower to
//    an `hc.const` wrapping the dtype's `TypeAttr`, usable both as an
//    argument (`x.astype(np.float32)`) and as a value-constructor
//    callee (`np.float16(0)`);
//  * most produced values get `!hc.undef` — type inference pins later;
//    kernel group parameters and launch-geometry query results are the
//    exceptions because launch metadata and internal `$` symbols are known
//    here.
//
// Intrinsic bodies are discarded. `@kernel.intrinsic`-decorated Python
// bodies are simulator fallbacks with no compilation meaning; target
// lowering recipes ride alongside as a sibling top-level `builtin.module`
// (`@__hc_intrinsic_lowerings__`) carrying a `transform.named_sequence`
// per `(intrinsic, target)` pair. That sibling is not an `hc_front.*` op,
// so this pass leaves it untouched — the eventual interpreter pass picks
// the recipes up by walking the tagged module. The lowered `hc.intrinsic`
// is a declaration (signature + scope/effects/const_kwargs + empty entry
// block with param args, zero body ops — no `hc.assign` either, since
// there is no body scan downstream for them to seed). A consequence worth
// spelling out: this pass does *not* validate the contents of an
// intrinsic body. Malformed ops inside a simulator fallback pass through
// as-is until the source op is erased.
//
// Explicitly deferred to later passes:
//  * loop-carried iter_arg analysis (`-hc-promote-names`).
//
// Two upstream frontend passes scrub the `hc_front` ghost ops the
// Python driver emits for source-level patterns that don't survive
// to `hc`. The canonical pipeline order is:
//
//     -hc-front-fold-region-defs -hc-front-inline \
//       -convert-hc-front-to-hc -hc-promote-names -hc-infer-types,
//       then canonicalization + CSE
//
// - `-hc-front-inline` expands every `hc_front.call` targeting a
//   `ref = {kind = "inline"}` marker func into an
//   `hc_front.inlined_region`. This pass consumes the region by
//   flattening it into the caller's block with a per-site alpha-
//   renamed prefix. A surviving `ref.kind = "inline"` call trips the
//   `run -hc-front-inline before -convert-hc-front-to-hc` diagnostic
//   in `lowerCall`.
// - `-hc-front-fold-region-defs` erases the ghost
//   `hc_front.name {ref.kind = "local"} + hc_front.call`
//   (+ optional `hc_front.return`) trail the frontend emits next to
//   a `@group.workitems` / `@group.subgroups` region for an
//   immediate-call shape (`inner()` / `return inner()`). The region
//   op itself is the lowering; the trail is dead. A surviving
//   `ref.kind = "local"` callee trips the
//   `run -hc-front-fold-region-defs before -convert-hc-front-to-hc`
//   diagnostic in `lowerCall`.
//
// Both diagnostics are worded as pipeline-ordering errors rather
// than "unsupported" so the operator's next step is obvious.

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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
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
// (0..<launch_rank). When launch metadata is absent, this is only an
// allocation cap for hand-written IR, not the semantic default rank.
constexpr int64_t kMaxLaunchAxis = 32;

struct LaunchMetadataAttrs {
  ArrayAttr workShape;
  ArrayAttr groupShape;
  IntegerAttr subgroupSize;

  bool empty() const { return !workShape && !groupShape && !subgroupSize; }
};

static FailureOr<Type> launchGeometryIdxType(MLIRContext *ctx, Location loc,
                                             StringRef prefix, unsigned axis) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallString<32> text(prefix);
  text += Twine(axis).str();
  std::string diag;
  FailureOr<sym::ExprHandle> handle = sym::parseExpr(store, text, &diag);
  if (failed(handle)) {
    emitError(loc) << "failed to synthesize launch-geo symbol '" << text
                   << "': " << diag;
    return failure();
  }
  return Type(IdxType::get(ctx, ExprAttr::get(ctx, *handle)));
}

static FailureOr<SmallVector<Type>> launchGeometryIdxTypes(MLIRContext *ctx,
                                                           Location loc,
                                                           StringRef prefix,
                                                           unsigned count) {
  SmallVector<Type> types;
  types.reserve(count);
  for (unsigned axis = 0; axis < count; ++axis) {
    FailureOr<Type> type = launchGeometryIdxType(ctx, loc, prefix, axis);
    if (failed(type))
      return failure();
    types.push_back(*type);
  }
  return types;
}

//===----------------------------------------------------------------------===//
// The Python driver (see `hc/_resolve.py`) stamps every `hc_front.name` and
// most `hc_front.attr` with a `ref` DictAttr of the form
//   {kind = "<class>", ...per-class payload...}
// `RefInfo` is the null-safe, minimally-typed view: it's cheap to construct,
// survives a missing `ref` on hand-written IR, and centralizes attribute
// lookups so every call site goes through the same code path.
//===----------------------------------------------------------------------===//

class RefInfo {
public:
  // Build from a defining op. Absent `ref` dict and dict-without-kind are
  // both modeled as "no classification" (empty kind); callers compare
  // `getKind()` against the expected string.
  static RefInfo get(Operation *op) {
    RefInfo info;
    if (!op)
      return info;
    info.dict_ = op->getAttrOfType<DictionaryAttr>("ref");
    if (info.dict_) {
      if (auto k = info.dict_.getAs<StringAttr>("kind"))
        info.kind_ = k.getValue();
    }
    return info;
  }

  // Build from a bare `DictionaryAttr` that already carries the
  // ref-shaped payload — used for parameter-side sub-dicts (e.g. the
  // captured `layout=` payload on an `hc_front.kernel` parameter
  // entry) that share the body-level ref schema but don't live on a
  // dedicated op of their own.
  static RefInfo fromDict(DictionaryAttr dict) {
    RefInfo info;
    info.dict_ = dict;
    if (dict) {
      if (auto k = dict.getAs<StringAttr>("kind"))
        info.kind_ = k.getValue();
    }
    return info;
  }

  // True iff a `ref` dict was present on the op.
  explicit operator bool() const { return static_cast<bool>(dict_); }

  // The `kind` payload. Empty when no `ref` or no `kind` string — callers
  // that need to distinguish "missing ref" from "present but malformed"
  // should check `bool(info) && getKind().empty()`.
  StringRef getKind() const { return kind_; }

  // String-valued key lookup. Returns empty on missing dict, missing key,
  // or non-string value.
  StringRef getString(StringRef key) const {
    if (!dict_)
      return {};
    if (auto s = dict_.getAs<StringAttr>(key))
      return s.getValue();
    return {};
  }

  // Typed attribute lookup. Returns a default-constructed (null) `AttrT`
  // when the dict is absent, the key is missing, or the value is not of
  // the requested type — mirrors `DictionaryAttr::getAs` exactly.
  template <typename AttrT> AttrT getAs(StringRef key) const {
    if (!dict_)
      return {};
    return dict_.getAs<AttrT>(key);
  }

  // Driver-contract check: `ref` is how the Python driver tells the pass
  // "this name/attr is a <kind>". A dict with no string `kind` is a
  // driver bug, not a fallback — return failure so every call site that
  // consumes `ref` can stop short rather than dropping into a misleading
  // "unsupported" path downstream. Absent `ref` is fine: hand-written IR
  // uses that to mean "no classification" and different consumers handle
  // it differently. The op mnemonic is already prepended by `emitOpError`,
  // so the caller doesn't need to pass a role string.
  LogicalResult diagnoseIfMalformed(Operation *op) const {
    if (!dict_ || !kind_.empty())
      return success();
    return op->emitOpError()
           << "has a `ref` dict with missing or non-string `kind`; the "
              "driver must populate a classification before this pass runs";
  }

private:
  DictionaryAttr dict_;
  StringRef kind_;
};

//===----------------------------------------------------------------------===//
// Attribute-conversion helpers.
//===----------------------------------------------------------------------===//

// Turn a plain builtin `["M + 1", ...]` string-array attribute (the form
// `hc_front` uses for `work_shape` / `group_shape`) into a `#hc.shape<...>`
// attribute. Returns null on a malformed dimension so the caller can emit a
// diagnostic against the source op.
static FailureOr<ShapeAttr> stringArrayToShape(Operation *sourceOp,
                                               ArrayAttr array) {
  MLIRContext *ctx = sourceOp->getContext();
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

static FailureOr<ExprAttr> stringToExpr(Operation *sourceOp, StringAttr text,
                                        StringRef attrName) {
  MLIRContext *ctx = sourceOp->getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  std::string diag;
  FailureOr<sym::ExprHandle> handle =
      sym::parseExpr(store, text.getValue(), &diag);
  if (failed(handle)) {
    sourceOp->emitOpError("failed to parse ")
        << attrName << " '" << text.getValue() << "': " << diag;
    return failure();
  }
  return ExprAttr::get(ctx, *handle);
}

static bool declaresNoneReturn(Operation *op) {
  auto returns = op->getAttrOfType<StringAttr>("returns");
  return returns && returns.getValue() == "None";
}

static bool unresolvedFrontFuncDeclaresNoResults(Operation *anchor,
                                                 StringRef callee) {
  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return false;

  for (Operation &op : *module.getBody()) {
    auto func = dyn_cast<hc_front::FuncOp>(&op);
    if (func && func.getName() == callee)
      return declaresNoneReturn(&op);
  }
  return false;
}

// The front-end emits `effects = "pure"` / `"read"` / `"write"` /
// `"read_write"` as a plain string. Translate into the typed `HC_EffectsAttr`
// that `hc.func` / `hc.intrinsic` want.
static std::optional<EffectClass> parseEffectClass(StringRef text) {
  return llvm::StringSwitch<std::optional<EffectClass>>(text)
      .Case("pure", EffectClass::Pure)
      .Case("read", EffectClass::Read)
      .Case("write", EffectClass::Write)
      .Case("read_write", EffectClass::ReadWrite)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// Numpy dtype strings -> MLIR builtin scalar types. The Python resolver
// (`_numpy_dtype_name` in `hc/_resolve.py`) classifies any live numpy
// scalar type as `ref = {kind = "numpy_dtype_type", dtype = "<name>"}`
// using the identifier numpy itself exposes. This pass supports a
// curated subset — the fixed-width scalars plus their common size-
// aliases. Signed aliases land as builtin `si<N>` and unsigned aliases land
// as builtin `ui<N>`, so downstream passes can recover the user's dtype
// intent instead of guessing from a signless width. Anything outside the set
// returns `nullopt` so the caller can surface a located diagnostic instead of
// fabricating an arbitrary type.
//===----------------------------------------------------------------------===//

static std::optional<Type> resolveNumpyDtypeType(MLIRContext *ctx,
                                                 StringRef name) {
  return llvm::StringSwitch<std::optional<Type>>(name)
      .Cases({"float16", "half"}, Float16Type::get(ctx))
      .Cases({"float32", "single"}, Float32Type::get(ctx))
      .Cases({"float64", "double"}, Float64Type::get(ctx))
      .Case("int8", IntegerType::get(ctx, 8, IntegerType::Signed))
      .Case("uint8", IntegerType::get(ctx, 8, IntegerType::Unsigned))
      .Case("int16", IntegerType::get(ctx, 16, IntegerType::Signed))
      .Case("uint16", IntegerType::get(ctx, 16, IntegerType::Unsigned))
      .Cases({"int32", "intc"}, IntegerType::get(ctx, 32, IntegerType::Signed))
      .Cases({"uint32", "uintc"},
             IntegerType::get(ctx, 32, IntegerType::Unsigned))
      .Cases({"int64", "intp", "longlong"},
             IntegerType::get(ctx, 64, IntegerType::Signed))
      .Cases({"uint64", "uintp", "ulonglong"},
             IntegerType::get(ctx, 64, IntegerType::Unsigned))
      .Cases({"bool", "bool_"}, IntegerType::get(ctx, 1))
      .Default(std::nullopt);
}

// Coerce the payload of a bare numeric `hc.const` (what Python integer /
// float literals lower to) into a typed scalar `IntegerAttr`/`FloatAttr`
// of `targetTy`. Used by the `np.<dtype>(lit)` value-constructor path,
// where the dtype handle picked up by `lowerAttr` is authoritative for
// the destination type and the positional literal supplies the value.
// `BoolAttr` is itself an `IntegerAttr` subclass in upstream MLIR, so
// `dyn_cast<IntegerAttr>` picks it up for the i1 case. Returns null if
// the source attribute isn't a scalar numeric (or is outside the set of
// safely-representable values for the target); callers fall back to the
// TypeAttr form so the downstream verifier (not this helper) produces
// the diagnostic about an unexpected payload. Fixed-width integer targets
// use APInt truncation so unsigned NumPy constructors retain their type and
// Python integer literals wrap to the target width.
//
// Float -> float and int -> float route through `APFloat::get(double)`
// inside `FloatAttr::get`, which tolerates NaN/Inf at any target
// precision. Float -> int is the tricky direction: a plain
// `static_cast<int64_t>(double)` is UB for NaN/Inf and for values
// outside `[INT64_MIN, INT64_MAX]` (C++ [conv.fpint]). Use hex-float
// literals of +/-2^63 (both exactly representable as `double`) to
// bracket the safe range, and treat i1 separately as NumPy's `bool_`
// truthiness rather than bit-pattern truncation — `APInt(1, 2)` would
// store 0 (low bit), flipping the user's boolean under our feet.
// Coerce `src` to the target float type. Float -> Float keeps the
// double precision, Int -> Float widens via `int64 -> double`.
static Attribute coerceNumpyLiteralToFloat(FloatType ft, Attribute src) {
  if (auto f = dyn_cast<FloatAttr>(src))
    return FloatAttr::get(ft, f.getValueAsDouble());
  if (auto i = dyn_cast<IntegerAttr>(src))
    return FloatAttr::get(ft, static_cast<double>(i.getInt()));
  return {};
}

// Float -> Integer for non-`i1` widths. Rejects non-finite or
// out-of-range doubles (the user's literal is invalid in the target
// width); otherwise sext/trunc-to-width via APInt.
static Attribute coerceFloatToWideInteger(IntegerType it, double v) {
  if (!std::isfinite(v) || v < -0x1.0p63 || v >= 0x1.0p63)
    return {};
  APInt bits(64, static_cast<uint64_t>(static_cast<int64_t>(v)),
             /*isSigned=*/true);
  return IntegerAttr::get(it, bits.sextOrTrunc(it.getWidth()));
}

// Coerce `src` to the target integer type. `i1` follows truthiness
// semantics (any nonzero is `1`); wider widths sign-extend / truncate
// the bit pattern.
static Attribute coerceNumpyLiteralToInteger(IntegerType it, Attribute src) {
  bool isI1 = it.getWidth() == 1;
  if (auto i = dyn_cast<IntegerAttr>(src)) {
    if (isI1)
      return IntegerAttr::get(it, i.getValue().isZero() ? 0 : 1);
    return IntegerAttr::get(it, i.getValue().sextOrTrunc(it.getWidth()));
  }
  if (auto f = dyn_cast<FloatAttr>(src)) {
    double v = f.getValueAsDouble();
    if (isI1)
      return IntegerAttr::get(it, v != 0.0 ? 1 : 0);
    return coerceFloatToWideInteger(it, v);
  }
  return {};
}

static Attribute coerceNumpyLiteral(Type targetTy, Attribute src) {
  if (auto ft = dyn_cast<FloatType>(targetTy))
    return coerceNumpyLiteralToFloat(ft, src);
  if (auto it = dyn_cast<IntegerType>(targetTy))
    return coerceNumpyLiteralToInteger(it, src);
  return {};
}

//===----------------------------------------------------------------------===//
// `group.load(a[row_sl, col_sl], shape=(M, K))` — the WMMA pattern —
// has to land as `hc.load %a[%row_sl, %col_sl], shape %shape`, not as a
// chained `hc.buffer_view` + zero-index `hc.load`. This helper peels
// the `hc.buffer_view` produced by `lowerSubscript` on the handle of
// load/vload/store when the caller didn't supply trailing positional
// indices of its own (mixing the view's own index list with caller-
// supplied positionals would be ambiguous). The `hc.buffer_view` is
// `Pure`; if it has no other users, DCE drops it later.
//
// Single-level peel only. Nested `hc.buffer_view`s arise from chained
// Python subscripts (`a[i][j]`), and by `hc.buffer_view`'s Python-like
// semantics the inner view applies to the outer-view's *leading* axis
// — not the original buffer's next axis. Splicing two index lists
// together would silently misaddress in the general slice/slice case,
// so we stop at one level and let the caller diagnose.
static Value peelBufferView(Value handle,
                            SmallVectorImpl<Value> &extraIndices) {
  if (!extraIndices.empty())
    return handle;
  auto view = handle.getDefiningOp<HCBufferViewOp>();
  if (!view)
    return handle;
  extraIndices.assign(view.getIndices().begin(), view.getIndices().end());
  return view.getBuffer();
}

//===----------------------------------------------------------------------===//
// Binary-op mnemonic table. `hc_front.binop "Add"(a, b)` spells out the
// Python AST's BinOp.op class name; the lowering picks the matching `hc`
// op and emits it against the two already-lowered operands. All results
// use `!hc.undef`, consistent with the rest of the pipeline's progressive
// typing.
//
// `Pow` is the structural-only entry — the front pass emits the carrier
// `hc.pow` and `-hc-lower-pow` does the unfold + diagnostic; no shape
// checking happens here so the diagnostic surface is single-sourced.
//===----------------------------------------------------------------------===//

static Value emitBinop(OpBuilder &builder, Location loc, StringRef kind,
                       Value lhs, Value rhs, Type undef, Operation *sourceOp) {
  if (kind == "Add")
    return HCAddOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Sub")
    return HCSubOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Mult")
    return HCMulOp::create(builder, loc, undef, lhs, rhs);
  // Both Python `/` (Div) and `//` (FloorDiv) route to `hc.div`, whose
  // ODS summary is "integer/float division (Python `//` for ints)": int
  // operands floor, float operands do true division. That collapses
  // Python's `/`-on-ints (true division returning float) into `//`-style
  // floor — an intentional compromise pre-inference. If a later pass
  // wants strict Python `/` semantics it needs a dedicated truediv op;
  // only this branch has to change.
  if (kind == "FloorDiv" || kind == "Div")
    return HCDivOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Mod")
    return HCModOp::create(builder, loc, undef, lhs, rhs);
  if (kind == "Pow")
    return HCPowOp::create(builder, loc, undef, lhs, rhs);
  sourceOp->emitOpError("unsupported hc_front.binop kind '") << kind << "'";
  return nullptr;
}

static FailureOr<ArrayAttr> parameterNamesFromDicts(ArrayAttr params,
                                                    Operation *sourceOp) {
  MLIRContext *ctx = sourceOp->getContext();
  SmallVector<Attribute> names;
  names.reserve(params.size());
  llvm::SmallDenseSet<StringRef> seen;
  for (auto [idx, param] : llvm::enumerate(params)) {
    auto dict = dyn_cast<DictionaryAttr>(param);
    if (!dict)
      return sourceOp->emitOpError(
                 "expected `parameters` entries to be DictAttr, got ")
             << param;
    auto name = dict.getAs<StringAttr>("name");
    if (!name)
      return sourceOp->emitOpError("`parameters` entry at index ")
             << idx << " missing `name` key";
    if (!seen.insert(name.getValue()).second)
      return sourceOp->emitOpError("duplicate parameter name '")
             << name.getValue() << "'";
    names.push_back(name);
  }
  return ArrayAttr::get(ctx, names);
}

static FailureOr<ArrayAttr>
keywordOnlyParametersFromDicts(ArrayAttr params, Operation *sourceOp) {
  MLIRContext *ctx = sourceOp->getContext();
  SmallVector<Attribute> keywordOnly;
  bool seenKeywordOnly = false;
  for (auto [idx, param] : llvm::enumerate(params)) {
    auto dict = dyn_cast<DictionaryAttr>(param);
    if (!dict)
      return sourceOp->emitOpError(
                 "expected `parameters` entries to be DictAttr, got ")
             << param;
    auto name = dict.getAs<StringAttr>("name");
    if (!name)
      return sourceOp->emitOpError("`parameters` entry at index ")
             << idx << " missing `name` key";
    auto passing = dict.getAs<StringAttr>("passing");
    if (!passing)
      return sourceOp->emitOpError("`parameters` entry at index ")
             << idx << " missing `passing` key";
    StringRef mode = passing.getValue();
    if (mode == "keyword_only") {
      seenKeywordOnly = true;
      keywordOnly.push_back(name);
      continue;
    }
    if (mode != "positional")
      return sourceOp->emitOpError("`parameters` entry '")
             << name.getValue() << "' has unsupported passing mode '" << mode
             << "'";
    if (seenKeywordOnly)
      return sourceOp->emitOpError("positional parameter '")
             << name.getValue() << "' cannot follow a keyword-only parameter";
  }
  return ArrayAttr::get(ctx, keywordOnly);
}

static LaunchMetadataAttrs launchMetadataAttrsFrom(Operation *op) {
  return {
      op->getAttrOfType<ArrayAttr>("work_shape"),
      op->getAttrOfType<ArrayAttr>("group_shape"),
      op->getAttrOfType<IntegerAttr>("subgroup_size"),
  };
}

struct LaunchMetadata {
  ShapeAttr workShape;
  ShapeAttr groupShape;
  ExprAttr subgroupSize;
};

static FailureOr<ExprAttr> subgroupSizeToExprAttr(Operation *sourceOp,
                                                  IntegerAttr attr) {
  if (!attr)
    return ExprAttr();
  MLIRContext *ctx = sourceOp->getContext();
  SmallString<32> text;
  attr.getValue().toStringSigned(text);
  std::string diagnostic;
  FailureOr<sym::ExprHandle> handle = sym::parseExpr(
      ctx->getOrLoadDialect<HCDialect>()->getSymbolStore(), text, &diagnostic);
  if (failed(handle)) {
    sourceOp->emitOpError("failed to parse subgroup_size expression: ")
        << diagnostic;
    return failure();
  }
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<LaunchMetadata>
parseLaunchMetadata(Operation *sourceOp, LaunchMetadataAttrs attrs) {
  LaunchMetadata metadata;
  FailureOr<ExprAttr> subgroupSize =
      subgroupSizeToExprAttr(sourceOp, attrs.subgroupSize);
  if (failed(subgroupSize))
    return failure();
  metadata.subgroupSize = *subgroupSize;
  if (attrs.workShape) {
    FailureOr<ShapeAttr> parsed = stringArrayToShape(sourceOp, attrs.workShape);
    if (failed(parsed))
      return failure();
    metadata.workShape = *parsed;
  }
  if (attrs.groupShape) {
    FailureOr<ShapeAttr> parsed =
        stringArrayToShape(sourceOp, attrs.groupShape);
    if (failed(parsed))
      return failure();
    metadata.groupShape = *parsed;
  }
  return metadata;
}

static void appendBoundSymbol(MLIRContext *ctx, StringRef name,
                              llvm::StringSet<> &seen,
                              SmallVectorImpl<Attribute> &symbols) {
  if (seen.insert(name).second)
    symbols.push_back(StringAttr::get(ctx, name));
}

static void appendExprBoundSymbols(MLIRContext *ctx, ExprAttr expr,
                                   llvm::StringSet<> &seen,
                                   SmallVectorImpl<Attribute> &symbols) {
  if (!expr)
    return;
  sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
    appendBoundSymbol(ctx, name, seen, symbols);
  });
}

static void appendPredBoundSymbols(MLIRContext *ctx, PredAttr pred,
                                   llvm::StringSet<> &seen,
                                   SmallVectorImpl<Attribute> &symbols) {
  if (!pred)
    return;
  sym::walkSymbolNames(pred.getValue(), [&](StringRef name) {
    appendBoundSymbol(ctx, name, seen, symbols);
  });
}

static void appendShapeBoundSymbols(MLIRContext *ctx, ShapeAttr shape,
                                    llvm::StringSet<> &seen,
                                    SmallVectorImpl<Attribute> &symbols) {
  if (!shape)
    return;
  for (Attribute dim : shape.getDims())
    appendExprBoundSymbols(ctx, dyn_cast<ExprAttr>(dim), seen, symbols);
}

// Free symbols inside a `#hc.layout<...>` payload that are *not*
// declared as layout-internal placeholders (`shape_syms`, `index_syms`,
// `params` keys) need to join the kernel's `bound_symbols` so the host
// wrapper / scope binder knows to materialize them at launch — for the
// default fully-strided buffer layout that's exactly the per-axis
// `$STRIDE_<N>_<argname>` symbols emitted in `parameterTypeFromDict`.
// Collect the names that bind locally to a layout — `shape_syms`,
// `index_syms`, and the `params` keys — into `layoutLocals`. These
// don't get promoted to kernel-level bound symbols.
static void collectLayoutLocalSymbolNames(LayoutAttr layout,
                                          llvm::StringSet<> &layoutLocals) {
  for (Attribute name : layout.getShapeSyms())
    if (auto str = dyn_cast<StringAttr>(name))
      layoutLocals.insert(str.getValue());
  for (Attribute name : layout.getIndexSyms())
    if (auto str = dyn_cast<StringAttr>(name))
      layoutLocals.insert(str.getValue());
  for (NamedAttribute kv : layout.getParams())
    layoutLocals.insert(kv.getName().getValue());
}

static void appendLayoutBoundSymbols(MLIRContext *ctx, LayoutAttr layout,
                                     llvm::StringSet<> &seen,
                                     SmallVectorImpl<Attribute> &symbols) {
  if (!layout)
    return;
  llvm::StringSet<> layoutLocals;
  collectLayoutLocalSymbolNames(layout, layoutLocals);

  auto walk = [&](ExprAttr expr) {
    if (!expr)
      return;
    sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
      if (layoutLocals.contains(name))
        return;
      appendBoundSymbol(ctx, name, seen, symbols);
    });
  };

  walk(layout.getStorageSize());
  walk(layout.getOffset());
  for (NamedAttribute kv : layout.getParams())
    if (auto expr = dyn_cast<ExprAttr>(kv.getValue()))
      walk(expr);
}

static void appendInputTypeBoundSymbols(MLIRContext *ctx, Type type,
                                        llvm::StringSet<> &seen,
                                        SmallVectorImpl<Attribute> &symbols) {
  if (auto buffer = dyn_cast<BufferType>(type)) {
    appendShapeBoundSymbols(ctx, buffer.getShape(), seen, symbols);
    appendLayoutBoundSymbols(ctx, buffer.getLayout(), seen, symbols);
    return;
  }
  if (auto idx = dyn_cast<IdxType>(type))
    return appendExprBoundSymbols(ctx, idx.getExpr(), seen, symbols);
  if (auto pred = dyn_cast<PredType>(type))
    return appendPredBoundSymbols(ctx, pred.getPred(), seen, symbols);
  if (auto tuple = dyn_cast<TupleType>(type))
    for (Type element : tuple.getTypes())
      appendInputTypeBoundSymbols(ctx, element, seen, symbols);
}

static void appendLaunchBoundSymbols(MLIRContext *ctx, StringRef prefix,
                                     unsigned rank, llvm::StringSet<> &seen,
                                     SmallVectorImpl<Attribute> &symbols) {
  for (unsigned axis = 0; axis < rank; ++axis) {
    SmallString<16> name(prefix);
    name += Twine(axis).str();
    appendBoundSymbol(ctx, name, seen, symbols);
  }
}

// Per-axis launch-geo literal bindings derived from a shape attr whose
// dims are integer-literal `#hc.expr`. Emits one `$<prefix><axis> ->
// IntegerAttr` entry per axis when the dim resolves to a literal; non-
// literal dims (symbolic kernel-arg names like `"W1"`) are skipped so
// the user-named symbols carry their own bindings from
// `hc.compile(symbols={...})`.
static void appendShapeLiteralBindings(MLIRContext *ctx, StringRef prefix,
                                       ShapeAttr shape,
                                       SmallVectorImpl<NamedAttribute> &out) {
  if (!shape)
    return;
  auto i64 = IntegerType::get(ctx, 64);
  for (auto [axis, dim] : llvm::enumerate(shape.getDims())) {
    auto expr = dyn_cast<ExprAttr>(dim);
    if (!expr)
      continue;
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()));
    if (!value)
      continue;
    SmallString<16> name(prefix);
    name += Twine(axis).str();
    out.emplace_back(StringAttr::get(ctx, name), IntegerAttr::get(i64, *value));
  }
}

// Augment the launcher-supplied `literal_bindings` dict with bindings
// derived from integer-literal launch-geo attrs on the front kernel:
// `group_shape` -> `$WGS<axis>`, `work_shape` -> `$WS<axis>`,
// `subgroup_size` -> `$WV0`, and the static product of `group_shape`
// -> `$GSZ0`. These are the launch-context symbols downstream passes
// (`hc-materialize-bound-exprs`, `hc-lower-launch-body`'s static-
// shape checks) expect to find substituted; user-supplied bindings
// (`hc.compile(symbols={...})`) carry the work-shape symbol names but
// can't reach the `$`-prefixed launch context, so the front-to-hc
// hand-off has to plant them eagerly. Duplicate user-supplied keys
// win to keep the launcher-provided values authoritative.
// Product of a shape's dims when every axis is an integer-literal
// `#hc.expr`. `std::nullopt` for the symbolic-mix case;
// `literal_bindings` carries IntegerAttr values only, so a symbolic
// product can't ride along as a binding regardless.
static std::optional<int64_t> staticShapeProduct(ShapeAttr shape) {
  if (!shape || shape.getDims().empty())
    return std::nullopt;
  int64_t product = 1;
  for (Attribute dim : shape.getDims()) {
    auto expr = dyn_cast<ExprAttr>(dim);
    if (!expr)
      return std::nullopt;
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()));
    if (!value)
      return std::nullopt;
    product *= *value;
  }
  return product;
}

// Merge `launcherBindings` with `derived`; user-supplied keys win
// (insertion order: launcher first, then derived entries whose names
// haven't been claimed yet). Returns the launcher dict unchanged
// when `derived` is empty.
static DictionaryAttr mergeLiteralBindings(MLIRContext *ctx,
                                           DictionaryAttr launcherBindings,
                                           ArrayRef<NamedAttribute> derived) {
  if (derived.empty())
    return launcherBindings;
  SmallVector<NamedAttribute> merged;
  llvm::StringSet<> seen;
  if (launcherBindings)
    for (NamedAttribute kv : launcherBindings) {
      seen.insert(kv.getName().getValue());
      merged.push_back(kv);
    }
  for (NamedAttribute kv : derived)
    if (seen.insert(kv.getName().getValue()).second)
      merged.push_back(kv);
  return DictionaryAttr::get(ctx, merged);
}

static DictionaryAttr augmentLiteralBindingsFromLaunchGeo(
    MLIRContext *ctx, DictionaryAttr launcherBindings, ShapeAttr workShape,
    ShapeAttr groupShape, IntegerAttr subgroupSize) {
  auto i64 = IntegerType::get(ctx, 64);
  SmallVector<NamedAttribute> derived;
  appendShapeLiteralBindings(ctx, "$WGS", groupShape, derived);
  appendShapeLiteralBindings(ctx, "$WS", workShape, derived);
  if (subgroupSize)
    derived.emplace_back(StringAttr::get(ctx, "$WV0"),
                         IntegerAttr::get(i64, subgroupSize.getInt()));
  if (auto product = staticShapeProduct(groupShape))
    derived.emplace_back(StringAttr::get(ctx, "$GSZ0"),
                         IntegerAttr::get(i64, *product));
  return mergeLiteralBindings(ctx, launcherBindings, derived);
}

static ArrayAttr buildKernelBoundSymbols(MLIRContext *ctx, TypeRange inputTypes,
                                         ShapeAttr workShape,
                                         ShapeAttr groupShape) {
  llvm::StringSet<> seen;
  SmallVector<Attribute> symbols;
  unsigned workRank =
      workShape ? static_cast<unsigned>(workShape.getDims().size()) : 0;
  unsigned groupRank =
      groupShape ? static_cast<unsigned>(groupShape.getDims().size()) : 0;
  unsigned groupIdRank = workRank ? workRank : groupRank;
  // `group_shape` aligns 1:1 with `work_shape` (`doc/langref.md`
  // "logical launch domain"). When the user only declares `work_shape`
  // and lets the runtime pick a default group_shape, the per-workgroup
  // launch-geo prefixes (`$WGS`, `$WI`, `$SGI`) still need to appear in
  // the kernel's bound-symbol table so `hc-materialize-bound-exprs`
  // recognises them as ambient. Mirror the rank fall-through in
  // `Lowerer::getLaunchGeometryRank` for the Workgroup domain.
  unsigned workgroupRank = groupRank ? groupRank : workRank;

  auto appendMethod = [&](LaunchGeoMethod method, unsigned rank) {
    appendLaunchBoundSymbols(ctx, getLaunchGeoMethodInfo(method).symbolPrefix,
                             rank, seen, symbols);
  };
  appendMethod(LaunchGeoMethod::GroupId, groupIdRank);
  appendMethod(LaunchGeoMethod::LocalId, workgroupRank);
  appendMethod(LaunchGeoMethod::SubgroupId, workgroupRank);
  appendMethod(LaunchGeoMethod::GroupShape, workgroupRank);
  appendMethod(LaunchGeoMethod::WorkOffset, workRank);
  appendMethod(LaunchGeoMethod::WorkShape, workRank);
  appendMethod(LaunchGeoMethod::GroupSize, 1);
  appendMethod(LaunchGeoMethod::WaveSize, 1);
  for (Type inputType : inputTypes)
    appendInputTypeBoundSymbols(ctx, inputType, seen, symbols);
  return ArrayAttr::get(ctx, symbols);
}

static std::optional<StringRef>
getLaunchContextParameterKind(DictionaryAttr param) {
  auto kind = param.getAs<StringAttr>("kind");
  if (!kind || kind.getValue() != "launch_context")
    return std::nullopt;
  auto context = param.getAs<StringAttr>("launch_context");
  if (!context)
    return StringRef();
  return context.getValue();
}

static std::optional<StringRef>
getScopedLaunchContextParameterKind(Operation *sourceOp) {
  auto scope = sourceOp->getAttrOfType<StringAttr>("scope");
  if (!scope)
    return std::nullopt;
  return llvm::StringSwitch<std::optional<StringRef>>(scope.getValue())
      .Case("WorkItem", StringRef("workitem"))
      .Case("SubGroup", StringRef("subgroup"))
      .Case("Subgroup", StringRef("subgroup"))
      .Default(std::nullopt);
}

static LogicalResult validateLaunchContextParameter(Operation *sourceOp,
                                                    DictionaryAttr param,
                                                    unsigned paramIndex,
                                                    StringRef actual,
                                                    StringRef expected) {
  auto name = param.getAs<StringAttr>("name");
  StringRef paramName = name ? name.getValue() : StringRef("<unknown>");
  if (actual.empty())
    return sourceOp->emitOpError("launch-context parameter '")
           << paramName << "' is missing string `launch_context`";
  if (paramIndex != 0)
    return sourceOp->emitOpError("launch-context parameter '")
           << paramName << "' must be the first parameter";
  if (!expected.empty() && actual != expected)
    return sourceOp->emitOpError("launch-context parameter '")
           << paramName << "' is '" << actual << "', expected '" << expected
           << "'";
  return success();
}

// Default fully-strided np/torch-style layout for a buffer kernel
// argument. Names per-axis stride symbols `$STRIDE_<axis>_<argname>`
// so the host wrapper / scope binder can match them against the
// runtime descriptor — `_mlir_ciface_hc_get_stride` is the eventual
// binding point. The structural form keeps `shape_syms` / `index_syms`
// as opaque placeholders (`d<i>` / `i<i>`) — the bound-name contract
// documented on `HC_LayoutAttr` only requires them to be unique and
// disjoint from `params` keys; the layout-flatten pass substitutes
// their identity at the use site.
//
// `storage_size` is informational for buffer layouts (the host owns the
// allocation; the verifier doesn't enforce `storage_size >= max(offset)
// + 1`), so emit a literal `0` and leave any precise bound for later
// passes to compute if they need it.
//
// Built structurally via `composeExpr*` so the resulting handles
// hash-cons against any other producer that builds the same expression
// — never via `parseExpr` / string templating per the symbolic-engine
// rules in `AGENTS.md`.
// Build the parallel `dN` (shape) and `iN` (index) symbol arrays for
// the default strided layout.
static void
buildDefaultStridedShapeIndexSyms(MLIRContext *ctx, unsigned rank,
                                  SmallVectorImpl<Attribute> &shapeSyms,
                                  SmallVectorImpl<Attribute> &indexSyms) {
  shapeSyms.reserve(rank);
  indexSyms.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    SmallString<8> shapeName("d");
    shapeName += Twine(i).str();
    SmallString<8> indexName("i");
    indexName += Twine(i).str();
    shapeSyms.push_back(StringAttr::get(ctx, shapeName));
    indexSyms.push_back(StringAttr::get(ctx, indexName));
  }
}

// Build a single per-axis `iN * $STRIDE_N_<arg>` term.
static FailureOr<sym::ExprHandle> composeStridedAxisTerm(sym::Store &store,
                                                         StringRef argName,
                                                         unsigned axis,
                                                         std::string &diag) {
  SmallString<8> indexName("i");
  indexName += Twine(axis).str();
  FailureOr<sym::ExprHandle> idx = sym::composeExprSym(store, indexName, &diag);
  if (failed(idx))
    return failure();
  SmallString<32> strideName("$STRIDE_");
  strideName += Twine(axis).str();
  strideName += "_";
  strideName += argName;
  FailureOr<sym::ExprHandle> stride =
      sym::composeExprSym(store, strideName, &diag);
  if (failed(stride))
    return failure();
  return sym::composeExprBinary(store, *idx, sym::ExprBinaryOp::Mul, *stride,
                                &diag);
}

// Compose the per-axis terms into a single offset expression
// `i0*$STRIDE_0_<arg> + i1*$STRIDE_1_<arg> + ...`. Rank 0 collapses
// to the literal `0`.
static FailureOr<sym::ExprHandle>
composeDefaultStridedOffset(sym::Store &store, StringRef argName, unsigned rank,
                            std::string &diag) {
  if (rank == 0)
    return sym::composeExprInt(store, 0, &diag);
  SmallVector<sym::ExprHandle> terms;
  terms.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    FailureOr<sym::ExprHandle> term =
        composeStridedAxisTerm(store, argName, axis, diag);
    if (failed(term))
      return failure();
    terms.push_back(*term);
  }
  sym::ExprHandle acc = terms[0];
  for (unsigned axis = 1; axis < rank; ++axis) {
    FailureOr<sym::ExprHandle> sum = sym::composeExprBinary(
        store, acc, sym::ExprBinaryOp::Add, terms[axis], &diag);
    if (failed(sum))
      return failure();
    acc = *sum;
  }
  return acc;
}

static FailureOr<LayoutAttr>
buildDefaultStridedBufferLayout(Operation *sourceOp, StringRef argName,
                                ShapeAttr shape) {
  MLIRContext *ctx = sourceOp->getContext();
  sym::Store &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  unsigned rank = static_cast<unsigned>(shape.getDims().size());

  SmallVector<Attribute> shapeSyms;
  SmallVector<Attribute> indexSyms;
  buildDefaultStridedShapeIndexSyms(ctx, rank, shapeSyms, indexSyms);

  auto composeFail = [&](StringRef what, StringRef diag) {
    return sourceOp->emitOpError("buffer parameter '")
           << argName << "': failed to compose " << what
           << " for default strided layout: " << diag;
  };

  std::string diag;
  FailureOr<sym::ExprHandle> offsetHandle =
      composeDefaultStridedOffset(store, argName, rank, diag);
  if (failed(offsetHandle))
    return composeFail("default strided offset", diag);

  FailureOr<sym::ExprHandle> storageSizeHandle =
      sym::composeExprInt(store, 0, &diag);
  if (failed(storageSizeHandle))
    return composeFail("storage_size literal", diag);

  return LayoutAttr::get(ctx, shapeSyms, indexSyms,
                         DictionaryAttr::get(ctx, {}),
                         ExprAttr::get(ctx, *storageSizeHandle),
                         ExprAttr::get(ctx, *offsetHandle));
}

// Read a Python-stamped ``layout`` ref dict and rebuild the structured
// ``#hc.layout<...>`` attribute. The resolver carries every piece as a
// typed MLIR attribute (`ArrayAttr<StringAttr>` for name lists,
// `DictionaryAttr` keyed on the param name with `ExprAttr` values,
// `ExprAttr` for `storage_size` / `offset`); see the resolver-side
// contract documented at `hc/_resolve.py::_index_map_ref`. No text is
// parsed here — assemble straight from the typed payload.
// Validate that every entry in `names` is a `StringAttr`. A bad entry
// is a frontend / resolver bug — surface with a localized diagnostic
// pointing at the offending index.
static LogicalResult validateLayoutSymNameArray(Operation *sourceOp,
                                                ArrayAttr names,
                                                StringRef which) {
  for (auto en : llvm::enumerate(names))
    if (!isa<StringAttr>(en.value())) {
      sourceOp->emitOpError("layout ref ")
          << which << " entry #" << en.index() << " is not a StringAttr (got "
          << en.value() << ")";
      return failure();
    }
  return success();
}

// Validate that every value in `params` is an `ExprAttr`.
static LogicalResult validateLayoutParamsDict(Operation *sourceOp,
                                              DictionaryAttr params) {
  for (NamedAttribute kv : params)
    if (!isa<ExprAttr>(kv.getValue())) {
      sourceOp->emitOpError("layout ref param '")
          << kv.getName().getValue() << "' is not an ExprAttr (got "
          << kv.getValue() << ")";
      return failure();
    }
  return success();
}

static FailureOr<LayoutAttr> layoutAttrFromRef(Operation *sourceOp,
                                               const RefInfo &ref) {
  auto require = [&](StringRef key, auto attr) -> LogicalResult {
    if (attr)
      return success();
    return sourceOp->emitOpError("layout ref missing `") << key << "`";
  };

  ArrayAttr shapeSymsAttr = ref.getAs<ArrayAttr>("shape_syms");
  ArrayAttr indexSymsAttr = ref.getAs<ArrayAttr>("index_syms");
  ExprAttr storageSize = ref.getAs<ExprAttr>("storage_size");
  ExprAttr offset = ref.getAs<ExprAttr>("offset");
  if (failed(require("shape_syms", shapeSymsAttr)) ||
      failed(require("index_syms", indexSymsAttr)) ||
      failed(require("storage_size", storageSize)) ||
      failed(require("offset", offset)))
    return failure();

  if (failed(
          validateLayoutSymNameArray(sourceOp, shapeSymsAttr, "shape_syms")) ||
      failed(validateLayoutSymNameArray(sourceOp, indexSymsAttr, "index_syms")))
    return failure();

  // Params is optional only in the "I have no derived params" sense
  // (default-strided builds a `DictionaryAttr::get(ctx, {})` of its own).
  // The Python resolver always emits the key — even empty — so a missing
  // entry here is a driver bug, not a parameterless layout.
  DictionaryAttr paramsAttr = ref.getAs<DictionaryAttr>("params");
  if (!paramsAttr)
    return sourceOp->emitOpError("layout ref missing `params`");
  if (failed(validateLayoutParamsDict(sourceOp, paramsAttr)))
    return failure();

  return LayoutAttr::get(sourceOp->getContext(),
                         llvm::to_vector(shapeSymsAttr.getValue()),
                         llvm::to_vector(indexSymsAttr.getValue()), paramsAttr,
                         storageSize, offset);
}

// Resolve the layout descriptor SSA argument fed to a `layout_op` call
// (currently `as_layout(value, descriptor)`). The descriptor must be
// produced by an `hc_front.name` whose `ref` was classified as
// `kind = "layout"`; anything else is a frontend bug — diagnose at the
// call site so users get the offending op rather than a downstream
// "missing attribute" complaint.
static FailureOr<LayoutAttr>
readLayoutFromValue(Value descriptor, Operation *consumer, StringRef role) {
  auto nameOp = descriptor.getDefiningOp<hc_front::NameOp>();
  if (!nameOp) {
    InFlightDiagnostic diag = consumer->emitOpError(role)
                              << " must be a captured IndexMap "
                                 "(hc_front.name reference)";
    if (Operation *def = descriptor.getDefiningOp())
      diag << "; got an SSA value defined by " << def->getName();
    else
      diag << "; got a block argument";
    return failure();
  }
  RefInfo ref = RefInfo::get(nameOp);
  if (failed(ref.diagnoseIfMalformed(nameOp)))
    return failure();
  if (ref.getKind() != "layout") {
    consumer->emitOpError(role)
        << " must be a captured IndexMap; got ref.kind '" << ref.getKind()
        << "'";
    return failure();
  }
  return layoutAttrFromRef(consumer, ref);
}

// Locate a specific keyword operand on a call op. Returns the
// `hc_front.keyword` op that names `kwname`, or null if absent. Keyword
// args are emitted as `hc_front.keyword "name" = %v` ops feeding the
// call's operand list, so a name match resolves uniquely. Callers want
// the keyword op (not just the underlying value) so they can chase
// classification metadata that lives on the source name op without
// going through `valueMap`, which intentionally maps non-SSA
// classifications (layout descriptors) to null.
static hc_front::KeywordOp findKeywordArg(hc_front::CallOp call,
                                          StringRef kwname) {
  for (Value arg : call.getArguments()) {
    auto k = arg.getDefiningOp<hc_front::KeywordOp>();
    if (k && k.getName() == kwname)
      return k;
  }
  return {};
}

// Consume the optional `layout=` keyword on a tensor-creation call,
// resolving the captured `IndexMap` descriptor to a `LayoutAttr`. Null
// `LayoutAttr` on success means "no `layout=` kwarg present" — the
// caller leaves the result untouched. `failure()` is a real error
// (malformed layout ref, wrong-kind descriptor, etc.) that has already
// emitted a diagnostic.
static FailureOr<LayoutAttr> consumeLayoutKwarg(hc_front::CallOp call) {
  hc_front::KeywordOp kw = findKeywordArg(call, "layout");
  if (!kw)
    return LayoutAttr();
  return readLayoutFromValue(kw.getValue(), call.getOperation(), "layout=");
}

// Build the explicit launch-context type (`!hc.group`, `!hc.workitem`,
// `!hc.subgroup`) from a `parameters` entry that carries a
// `launch_context` kind hint. Validates ordering / scope expectations
// before consulting the launch metadata.
static FailureOr<Type>
buildLaunchContextParameterType(Operation *sourceOp, DictionaryAttr param,
                                unsigned paramIndex, StringRef launchContext,
                                LaunchMetadataAttrs metadataAttrs) {
  MLIRContext *ctx = sourceOp->getContext();
  StringRef expected = "";
  if (auto scopedExpected = getScopedLaunchContextParameterKind(sourceOp))
    expected = *scopedExpected;
  if (failed(validateLaunchContextParameter(sourceOp, param, paramIndex,
                                            launchContext, expected)))
    return failure();
  FailureOr<LaunchMetadata> metadata =
      parseLaunchMetadata(sourceOp, metadataAttrs);
  if (failed(metadata))
    return failure();
  if (launchContext == "group")
    return Type(GroupType::get(ctx, metadata->workShape, metadata->groupShape,
                               metadata->subgroupSize));
  if (launchContext == "workitem")
    return Type(
        WorkitemType::get(ctx, metadata->groupShape, metadata->subgroupSize));
  if (launchContext == "subgroup")
    return Type(
        SubgroupType::get(ctx, metadata->groupShape, metadata->subgroupSize));
  return sourceOp->emitOpError("unknown launch-context parameter kind '")
         << launchContext << "'";
}

// Resolve the layout attribute for a buffer parameter — either the
// frontend-captured `IndexMap` ref dict or the default fully-strided
// layout namespaced by the arg name.
static FailureOr<LayoutAttr> resolveBufferParameterLayout(Operation *sourceOp,
                                                          DictionaryAttr param,
                                                          StringRef name,
                                                          ShapeAttr shape) {
  if (auto layoutDict = param.getAs<DictionaryAttr>("layout"))
    return layoutAttrFromRef(sourceOp, RefInfo::fromDict(layoutDict));
  return buildDefaultStridedBufferLayout(sourceOp, name, shape);
}

// Resolve a `kind = "buffer"` parameter dict to a `BufferType`. The
// fallback element type is used when the entry lacks an explicit
// `dtype` (e.g. dtype-polymorphic kernels).
static FailureOr<Type>
buildBufferParameterType(Operation *sourceOp, DictionaryAttr param,
                         StringRef name, ArrayAttr shapeAttr, Type fallback) {
  MLIRContext *ctx = sourceOp->getContext();
  Type elementType = fallback;
  if (auto dtype = param.getAs<StringAttr>("dtype")) {
    std::optional<Type> resolved = resolveNumpyDtypeType(ctx, dtype.getValue());
    if (!resolved)
      return sourceOp->emitOpError("buffer parameter '")
             << name << "' has unsupported dtype '" << dtype.getValue() << "'";
    elementType = *resolved;
  }

  FailureOr<ShapeAttr> shape = stringArrayToShape(sourceOp, shapeAttr);
  if (failed(shape))
    return failure();
  // Buffer args carry the default fully-strided np/torch layout from
  // the boundary on unless the Python frontend captured an explicit
  // `IndexMap` on the annotation (e.g. `Buffer[M, N, dtype, A_LAYOUT]`).
  // Per-axis stride symbols on the default-strided path are namespaced
  // by the arg name so two buffers with the same shape don't share
  // strides; the host wrapper binds them at launch.
  // `appendLayoutBoundSymbols` (called further down) picks up either
  // layout's free symbols and extends `kernel.bound_symbols` so
  // downstream passes know to leave them unmaterialized.
  FailureOr<LayoutAttr> layout =
      resolveBufferParameterLayout(sourceOp, param, name, *shape);
  if (failed(layout))
    return failure();
  return Type(BufferType::get(ctx, elementType, *shape, *layout));
}

// Implicit `group` parameter (legacy frontend-style top-level
// kernels): when the parameter name is literally `group` and we're
// not in a scoped helper, materialize the launch-context type from
// the kernel's launch metadata.
static FailureOr<Type>
buildImplicitGroupParameterType(Operation *sourceOp, StringAttr name,
                                LaunchMetadataAttrs metadataAttrs) {
  MLIRContext *ctx = sourceOp->getContext();
  if (name.getValue() != "group" ||
      getScopedLaunchContextParameterKind(sourceOp))
    return Type();
  FailureOr<LaunchMetadata> metadata =
      parseLaunchMetadata(sourceOp, metadataAttrs);
  if (failed(metadata))
    return failure();
  return Type(GroupType::get(ctx, metadata->workShape, metadata->groupShape,
                             metadata->subgroupSize));
}

// Validate the `kind` / `shape` keys for a non-launch-context, non-
// implicit-group parameter, returning either the fallback element
// type (no shape, plain scalar) or success() to indicate the buffer
// branch should be taken.
static FailureOr<Type> validateNonBufferParameterShape(Operation *sourceOp,
                                                       StringAttr name,
                                                       StringAttr kind,
                                                       ArrayAttr shapeAttr,
                                                       Type fallback) {
  if (!kind) {
    if (shapeAttr)
      return sourceOp->emitOpError("parameter '")
             << name.getValue()
             << "' has `shape` metadata but no string `kind`";
    return fallback;
  }
  if (kind.getValue() != "buffer") {
    if (shapeAttr)
      return sourceOp->emitOpError("parameter '")
             << name.getValue() << "' has `shape` metadata but kind '"
             << kind.getValue() << "' is not `buffer`";
    return fallback;
  }
  return Type();
}

// Once we've ruled out launch-context entries, the remaining parameter
// shapes are: implicit `group`, non-buffer with `shape` metadata coercion,
// or a buffer-with-layout. Walk those in the documented order so a
// missing `shape` only fires after the bespoke kinds have a chance to claim
// the parameter.
static FailureOr<Type> buildBufferLikeParameterType(
    Operation *sourceOp, DictionaryAttr param, StringAttr name, StringAttr kind,
    ArrayAttr shapeAttr, Type fallback, LaunchMetadataAttrs metadataAttrs) {
  FailureOr<Type> implicitGroup =
      buildImplicitGroupParameterType(sourceOp, name, metadataAttrs);
  if (failed(implicitGroup))
    return failure();
  if (*implicitGroup)
    return *implicitGroup;
  FailureOr<Type> nonBuffer = validateNonBufferParameterShape(
      sourceOp, name, kind, shapeAttr, fallback);
  if (failed(nonBuffer))
    return failure();
  if (*nonBuffer)
    return *nonBuffer;
  if (!shapeAttr)
    return sourceOp->emitOpError("buffer parameter '")
           << name.getValue() << "' is missing `shape` metadata";
  return buildBufferParameterType(sourceOp, param, name.getValue(), shapeAttr,
                                  fallback);
}

static FailureOr<Type>
parameterTypeFromDict(Operation *sourceOp, DictionaryAttr param, Type fallback,
                      LaunchMetadataAttrs defaultLaunchMetadata,
                      unsigned paramIndex) {
  auto kind = param.getAs<StringAttr>("kind");
  auto shapeAttr = param.getAs<ArrayAttr>("shape");
  auto name = param.getAs<StringAttr>("name");
  if (!name)
    return sourceOp->emitOpError("`parameters` entry missing `name` key");
  LaunchMetadataAttrs sourceMetadata = launchMetadataAttrsFrom(sourceOp);
  LaunchMetadataAttrs metadataAttrs =
      !sourceMetadata.empty() ? sourceMetadata : defaultLaunchMetadata;
  if (std::optional<StringRef> launchContext =
          getLaunchContextParameterKind(param))
    return buildLaunchContextParameterType(sourceOp, param, paramIndex,
                                           *launchContext, metadataAttrs);
  if (auto scopedExpected = getScopedLaunchContextParameterKind(sourceOp);
      scopedExpected && paramIndex == 0)
    return sourceOp->emitOpError("first scoped helper parameter must be "
                                 "marked as a ")
           << *scopedExpected << " launch context";
  return buildBufferLikeParameterType(sourceOp, param, name, kind, shapeAttr,
                                      fallback, metadataAttrs);
}

// Reject any `record` entry that is not in `allowedKeys`. Entries
// outside the spelt-out vocabulary are typo-likely and silently
// ignoring them would let the wrong shape pin a contract.
static LogicalResult verifyContractDictKeys(Operation *sourceOp,
                                            DictionaryAttr record,
                                            StringRef attrName, unsigned index,
                                            StringRef kind,
                                            ArrayRef<StringRef> allowedKeys) {
  llvm::SmallDenseSet<StringRef, 4> allowed(allowedKeys.begin(),
                                            allowedKeys.end());
  for (NamedAttribute attr : record) {
    StringRef key = attr.getName().getValue();
    if (!allowed.contains(key))
      return sourceOp->emitOpError("`")
             << attrName << "` entry #" << index << " kind '" << kind
             << "' has unsupported key '" << key << "'";
  }
  return success();
}

// Build a `kind = "idx"` contract entry. Optional `expr` is parsed
// once via the textual surface; absent expr means an unbound index.
static FailureOr<Type> buildContractIdxType(Operation *sourceOp,
                                            DictionaryAttr record,
                                            StringRef attrName, unsigned index,
                                            StringRef kind) {
  MLIRContext *ctx = sourceOp->getContext();
  if (failed(verifyContractDictKeys(sourceOp, record, attrName, index, kind,
                                    {"kind", "expr"})))
    return failure();
  if (auto expr = record.getAs<StringAttr>("expr")) {
    FailureOr<ExprAttr> parsed = stringToExpr(sourceOp, expr, "idx expr");
    if (failed(parsed))
      return failure();
    return Type(IdxType::get(ctx, *parsed));
  }
  return Type(IdxType::get(ctx, ExprAttr()));
}

// Build a `kind = "tensor"` or `kind = "vector"` contract entry.
// Both share the `{kind, shape, dtype}` vocabulary; the chosen
// constructor differentiates between the two.
static FailureOr<Type> buildContractShapedType(Operation *sourceOp,
                                               DictionaryAttr record,
                                               StringRef attrName,
                                               unsigned index, StringRef kind,
                                               bool isTensor) {
  MLIRContext *ctx = sourceOp->getContext();
  if (failed(verifyContractDictKeys(sourceOp, record, attrName, index, kind,
                                    {"kind", "shape", "dtype"})))
    return failure();

  auto dtype = record.getAs<StringAttr>("dtype");
  if (!dtype)
    return sourceOp->emitOpError("`")
           << attrName << "` entry #" << index << " missing string `dtype`";
  std::optional<Type> elementType =
      resolveNumpyDtypeType(ctx, dtype.getValue());
  if (!elementType)
    return sourceOp->emitOpError("`")
           << attrName << "` entry #" << index << " has unsupported dtype '"
           << dtype.getValue() << "'";
  auto shapeAttr = record.getAs<ArrayAttr>("shape");
  if (!shapeAttr)
    return sourceOp->emitOpError("`")
           << attrName << "` entry #" << index << " missing `shape`";
  FailureOr<ShapeAttr> shape = stringArrayToShape(sourceOp, shapeAttr);
  if (failed(shape))
    return failure();
  if (isTensor)
    return Type(::mlir::hc::TensorType::get(ctx, *elementType, *shape));
  return Type(::mlir::hc::VectorType::get(ctx, *elementType, *shape));
}

static FailureOr<Type> typeFromContractDict(Operation *sourceOp,
                                            DictionaryAttr record,
                                            StringRef attrName,
                                            unsigned index) {
  MLIRContext *ctx = sourceOp->getContext();
  auto kind = record.getAs<StringAttr>("kind");
  if (!kind)
    return sourceOp->emitOpError("`")
           << attrName << "` entry #" << index << " missing string `kind`";
  StringRef kindStr = kind.getValue();
  if (kindStr == "undef") {
    if (failed(verifyContractDictKeys(sourceOp, record, attrName, index,
                                      kindStr, {"kind"})))
      return failure();
    return Type(UndefType::get(ctx));
  }
  if (kindStr == "idx")
    return buildContractIdxType(sourceOp, record, attrName, index, kindStr);
  if (kindStr == "tensor")
    return buildContractShapedType(sourceOp, record, attrName, index, kindStr,
                                   /*isTensor=*/true);
  if (kindStr == "vector")
    return buildContractShapedType(sourceOp, record, attrName, index, kindStr,
                                   /*isTensor=*/false);
  return sourceOp->emitOpError("`")
         << attrName << "` entry #" << index << " has unsupported kind '"
         << kindStr << "'";
}

static FailureOr<SmallVector<Type>> typesFromContractArray(Operation *sourceOp,
                                                           ArrayAttr array,
                                                           StringRef attrName) {
  SmallVector<Type> types;
  types.reserve(array.size());
  for (auto [index, item] : llvm::enumerate(array)) {
    auto record = dyn_cast<DictionaryAttr>(item);
    if (!record)
      return sourceOp->emitOpError("`") << attrName << "` entry #" << index
                                        << " must be a dictionary attribute";
    FailureOr<Type> type =
        typeFromContractDict(sourceOp, record, attrName, index);
    if (failed(type))
      return failure();
    types.push_back(*type);
  }
  return types;
}

//===----------------------------------------------------------------------===//
// Lowerer. Instantiated once per top-level `hc_front` callable; walks the
// body, threads the scope map through nested regions, and emits the
// corresponding `hc` ops via the shared `builder`. Source ops are erased
// after each top-level op finishes lowering.
//===----------------------------------------------------------------------===//

// Call-site special-cased kwargs get picked out of the argument list
// before operands are lowered to real `hc` ops, so the callee can see
// a flat positional arg list plus a {name: attr} map. Lives at file
// scope (rather than nested inside `Lowerer`) so static helpers
// outside the class can take it by reference.
struct CallArgs {
  SmallVector<Value> positional;
  llvm::StringMap<Attribute> kwattrs;
  llvm::StringMap<Value> kwvalues;
};

class Lowerer {
public:
  Lowerer(OpBuilder &builder, Type undef,
          LaunchMetadataAttrs defaultLaunchMetadata)
      : builder(builder), undef(undef),
        defaultLaunchMetadata(defaultLaunchMetadata) {}

  LogicalResult lowerCallable(Operation *frontOp);

  using BodyBuilder =
      llvm::function_ref<FailureOr<Region *>(Block *, FunctionType)>;
  LogicalResult runCallableBody(Operation *frontOp, ArrayAttr runParams,
                                bool returnsValue, bool ensureReturn,
                                BodyBuilder build);
  LogicalResult lowerKernelCallable(hc_front::KernelOp kernel,
                                    ArrayAttr params);
  LogicalResult lowerFuncCallable(hc_front::FuncOp func, ArrayAttr params);
  LogicalResult lowerIntrinsicCallable(hc_front::IntrinsicOp intr,
                                       ArrayAttr params);
  LogicalResult populateKernelMetadata(HCKernelOp hcKernel, Operation *frontOp,
                                       FunctionType fnType);
  FailureOr<FunctionType> buildIntrinsicFunctionType(Operation *frontOp,
                                                     ArrayAttr parameterNames,
                                                     ArrayAttr constKwargsAttr);

private:
  OpBuilder &builder;
  Type undef;
  LaunchMetadataAttrs defaultLaunchMetadata;

  // No scope stack: name-store placeholders carry the binding. Entry
  // points that introduce a new Python-level name (parameter entry,
  // for-loop IV, region parameters) emit `hc.assign` at their block
  // entries; every `hc_front.name` read lowers to `hc.name_load`. See
  // the `ConvertHCFrontToHC` ODS description for the full contract.

  // `hc_front.keyword "name" = %v` is consumed at the parent call. The
  // map is keyed by the keyword op's SSA result and stores the name, the
  // lowered value (may be null for attr operands that have no standalone hc
  // counterpart).
  struct KeywordInfo {
    StringRef name;
    Value loweredValue;
  };
  llvm::DenseMap<Value, KeywordInfo> keywordInfo;

  // Every SSA value produced inside the `hc_front` body maps to the `hc`
  // value that replaces it. This is the workhorse translation table used
  // by every operand lookup.
  llvm::DenseMap<Value, Value> valueMap;

  std::optional<unsigned> groupRank;
  std::optional<unsigned> workRank;
  ShapeAttr launchWorkShape;
  ShapeAttr launchGroupShape;
  ExprAttr launchSubgroupSize;
  llvm::DenseMap<Value, llvm::StringMap<unsigned>> staticLaunchGeoRanks;

  // Resolves a classified `hc_front.name`. Classifier kinds that name a
  // Python-level binding (`param`/`local`/`iv`) lower to an
  // `hc.name_load "<ident>"` — a placeholder the promotion pass replaces
  // with a direct SSA use. Constant and symbol kinds materialize a real
  // `hc` producer eagerly.
  // Returns:
  //   * `success(Value)`  — a usable hc value for the name;
  //   * `success(Value())` — the name is consumed at the call site and
  //                          intentionally has no SSA counterpart (e.g.
  //                          callee/intrinsic/inline/builtin/module refs);
  //   * `failure()`        — a diagnostic was emitted.
  FailureOr<Value> lowerName(hc_front::NameOp op);

  // Lowers `hc_front.attr`. Most attrs produce no standalone SSA value —
  // they are consumed by the parent call/subscript via the attr op's
  // `$name`. The one exception is `ref.kind = "numpy_dtype_type"`: these
  // materialize to an `hc.const` wrapping the dtype's `TypeAttr`, reused
  // by both the argument path (`x.astype(np.<dt>)`) and the callee path
  // (`np.<dt>(0)`). Returns `failure` only when the op's `ref` dict is
  // present-but-malformed.
  FailureOr<Value> lowerAttr(hc_front::AttrOp op);

  LogicalResult lowerRegion(Region &src);

  void collectStaticLaunchGeometryRanks(Operation *frontOp);
  std::optional<unsigned> getStaticLaunchGeometryRank(Value source,
                                                      StringRef method) const;

  // Shared body for `hc.workitem_region` / `hc.subgroup_region`. Both have
  // identical shape (captures attr + single-block body + optional
  // parameters stamped on the front op that become block args) so we
  // route through one helper templated on the target `hc` op type.
  template <typename HCRegionOpT, typename FrontRegionOpT>
  LogicalResult lowerCapturingRegion(FrontRegionOpT op);
  template <typename FrontRegionOpT>
  LogicalResult populateCapturingRegionParams(
      FrontRegionOpT op, Operation *newOp, ArrayAttr params, Block *body,
      StringRef expectedLaunchContext, SmallVectorImpl<StringAttr> &paramNames);
  template <typename FrontRegionOpT>
  LogicalResult lowerCapturingRegionBody(FrontRegionOpT op, Operation *newOp,
                                         bool isTailReturnRegion);
  template <typename FrontRegionOpT>
  FailureOr<SmallVector<Type>>
  capturingRegionResultTypes(FrontRegionOpT op, bool isTailReturnRegion);

  LogicalResult lowerOp(Operation *op);
  LogicalResult lowerProducingOp(Operation *op);
  LogicalResult lowerScalarValueOp(Operation *op);
  LogicalResult lowerStructuralOp(Operation *op);
  LogicalResult lowerTupleOp(hc_front::TupleOp t);
  LogicalResult lowerKeywordOp(hc_front::KeywordOp k);

  FailureOr<Value> lowerValueOperand(Value v, Operation *consumer,
                                     StringRef role = "operand");
  FailureOr<SmallVector<Value>> lowerValueOperands(ValueRange vs,
                                                   Operation *consumer,
                                                   StringRef role = "operand");
  FailureOr<SmallVector<Value>> expandTupleOperand(Value v, Operation *consumer,
                                                   StringRef role);
  LogicalResult
  classifyFrontSubscriptIndex(Value frontIdx, hc_front::SubscriptOp op,
                              SmallVectorImpl<Value> &residualIndices,
                              SmallVectorImpl<int64_t> &unitAxes,
                              size_t &outputPos);
  FailureOr<SmallVector<Value>> lowerReturnValues(hc_front::ReturnOp op);

  // Per-op lowering entry points. `Value`-returning variants write into
  // `valueMap`; `LogicalResult`-returning ones are terminators or
  // side-effecting ops without a user-visible result.
  Value lowerConstant(hc_front::ConstantOp op);
  Value lowerBinop(hc_front::BinOp op);
  Value lowerSlice(hc_front::SliceOp op);
  LogicalResult lowerReturn(hc_front::ReturnOp op);
  LogicalResult lowerAssign(hc_front::AssignOp op);
  LogicalResult lowerAssignToName(hc_front::AssignOp op,
                                  hc_front::TargetNameOp tn);
  LogicalResult lowerAssignToTuple(hc_front::AssignOp op,
                                   hc_front::TargetTupleOp tt);
  LogicalResult lowerAssignToSubscript(hc_front::AssignOp op,
                                       hc_front::TargetSubscriptOp ts);
  FailureOr<SmallVector<Value>>
  expandTupleAssignSources(hc_front::AssignOp op, Value value, size_t arity);
  LogicalResult lowerFor(hc_front::ForOp op);
  FailureOr<hc_front::CallOp> lowerForIterRegion(hc_front::ForOp op,
                                                 Region &iter);
  FailureOr<std::array<Value, 3>> lowerForRangeArgs(hc_front::ForOp op,
                                                    hc_front::CallOp iterCall);
  LogicalResult lowerWorkitemRegion(hc_front::WorkitemRegionOp op);
  LogicalResult lowerSubgroupRegion(hc_front::SubgroupRegionOp op);
  // Flattens a `hc_front.inlined_region` into the caller's current `hc`
  // insertion block. The region is a pure name-scope boundary stamped
  // by `-hc-front-inline`; consuming it here keeps the `hc` output
  // free of `hc_front` artifacts. Alpha-renames every local name in
  // the cloned body with a per-site prefix, emits `hc.assign` for each
  // parameter binding, walks the body through `lowerOp`, and intercepts
  // `hc_front.return` to wire the region's results into `valueMap`.
  LogicalResult lowerInlinedRegion(hc_front::InlinedRegionOp op);
  LogicalResult bindInlinedRegionParams(hc_front::InlinedRegionOp op,
                                        ArrayAttr params, StringRef prefix);
  FailureOr<SmallVector<Value>>
  lowerInlinedRegionBody(hc_front::InlinedRegionOp op, Block &block);
  LogicalResult publishInlinedRegionResults(hc_front::InlinedRegionOp op,
                                            ArrayRef<Value> resultValues);
  // FailureOr so a `builtin` call that is cleanly consumed by its parent
  // (`range(...)` inside `hc_front.for`) can be distinguished from a real
  // error. `FailureOr<Value>` reads as: success+Value / success+null /
  // failure.
  FailureOr<Value> lowerCall(hc_front::CallOp op);
  FailureOr<Value> lowerNamedCall(hc_front::CallOp op, RefInfo ref,
                                  StringRef kind, const CallArgs &args);
  FailureOr<Value> lowerCalleeCall(hc_front::CallOp op,
                                   FlatSymbolRefAttr symRef, StringRef callee,
                                   const CallArgs &args);
  FailureOr<Value> lowerIntrinsicCall(hc_front::CallOp op,
                                      FlatSymbolRefAttr symRef,
                                      StringRef callee, const CallArgs &args);
  SmallVector<Type> deduceCalleeResultTypes(hc_front::CallOp op,
                                            FlatSymbolRefAttr symRef,
                                            StringRef callee);
  Value lowerSubscript(hc_front::SubscriptOp op);
  struct SubscriptFoldResult {
    bool consumed;
    Value value;
  };
  SubscriptFoldResult trySubscriptFolds(hc_front::SubscriptOp op);
  Value lowerGenericSubscript(hc_front::SubscriptOp op);
  FailureOr<Value> tryLowerAttrSubscript(hc_front::SubscriptOp op,
                                         hc_front::AttrOp attr);
  FailureOr<Value> tryLowerCallSubscript(hc_front::SubscriptOp op,
                                         hc_front::CallOp call);
  Value tryLowerShapeSubscript(hc_front::SubscriptOp op, Value baseVal,
                               IntegerAttr ax);
  FailureOr<Value> tryLowerLaunchGeoAttrSubscript(hc_front::SubscriptOp op,
                                                  hc_front::AttrOp attr,
                                                  Value idxVal, IntegerAttr ax);

  // Populates `entry`'s block arguments and returns the matching
  // `FunctionType`. Buffer parameters are seeded from their metadata;
  // everything else starts as `!hc.undef`. No `hc.assign` is emitted here;
  // param-name publishing happens in `emitParameterAssigns` once the
  // enclosing op has attached `entry` to its body.
  FailureOr<FunctionType> materializeParameters(ArrayAttr params, Block &entry,
                                                Operation *sourceOp,
                                                bool returnsValue);

  // Helper used by `materializeParameters` to enforce the scoped-helper
  // first-launch-context rule before any parameter types are computed.
  LogicalResult validateScopedHelperLaunchContexts(Operation *sourceOp,
                                                   ArrayAttr params);

  // Helper used by `materializeParameters` to mirror launch geometry off
  // the freshly-computed launch-context parameter type into the lowerer's
  // own group/work/subgroup state.
  void recordLaunchContextShape(Type paramType);

  // Emits `hc.assign "<pname>", %arg` at the start of `entry` for each
  // parameter in `params`. Must be called after the enclosing op has
  // taken ownership of `entry`, since `hc.assign` needs an insertion
  // point inside the live region. `params` is assumed pre-validated by
  // `materializeParameters` — this helper does not re-check the dict.
  // Positions the builder itself (caller is expected to hold an
  // `OpBuilder::InsertionGuard` if the builder state matters past this
  // call).
  void emitParameterAssigns(ArrayAttr params, Block &entry, Location loc);

  // Emit an `hc.const` for a `ref.kind = "constant"` name op. Needs the
  // python_kind / value payload because the front dialect packs it as
  // string-for-anything-but-int. `FailureOr` matches the `lowerName`
  // idiom: parse failures diagnose-and-fail, not return null.
  FailureOr<Value> emitConstantFromRef(Location loc, const RefInfo &ref,
                                       Operation *sourceOp);

  // Consume a `hc_front.call` dispatched to a `dsl_method` callee. The
  // FailureOr<Value> return mirrors `lowerCall`: failure = error emitted,
  // success+null = the call produced no SSA result (e.g. `x.store(...)`).
  FailureOr<Value> lowerDslMethodCall(hc_front::CallOp call,
                                      hc_front::AttrOp attr);

  FailureOr<CallArgs> collectCallArgs(hc_front::CallOp op);

  FailureOr<Value> lowerNumpyDtypeCall(hc_front::CallOp call,
                                       const RefInfo &ref,
                                       const CallArgs &args);
  FailureOr<Value> lowerUnaryBaseMethod(hc_front::CallOp call, StringRef method,
                                        Value base, const CallArgs &args);
  FailureOr<Value> lowerReduceMethod(hc_front::CallOp call, StringRef method,
                                     Value base, const CallArgs &args);
  // `np.<func>(...)` for the bench of NumPy ufuncs the front pass
  // forwards to `hc.builtin_call`. Called from `lowerDslMethodCall`
  // when the attr base is `numpy_attr` and the method passes
  // `isNumpyBuiltinCallMethod`. The returned value is a
  // `hc.builtin_call` with `name = "numpy.<method>"`; downstream
  // lowering pattern-matches on the name to emit `math.*` (or a
  // runtime call where needed).
  FailureOr<Value> lowerNumpyBuiltinCall(hc_front::CallOp call,
                                         StringRef method,
                                         const CallArgs &args);
  // Method-bucket dispatch for the value-base case (`x.vec()`,
  // `x.sum()`, `group.load(...)`, `group.shape`, ...). Split out of
  // `lowerDslMethodCall` so the outer routine stays a flat list of
  // ref-kind early returns and the per-bucket fan-out lives in one
  // place.
  FailureOr<Value> lowerValueBaseMethodCall(hc_front::CallOp call,
                                            hc_front::AttrOp attr,
                                            StringRef method,
                                            const CallArgs &args);
  FailureOr<Value> lowerMemOp(hc_front::CallOp call, StringRef method,
                              const CallArgs &args);
  FailureOr<Value> lowerMemLoad(hc_front::CallOp call, StringRef method,
                                const CallArgs &args);
  FailureOr<Value> lowerMemStore(hc_front::CallOp call, const CallArgs &args);
  FailureOr<Value> lowerMemInit(hc_front::CallOp call, StringRef method,
                                const CallArgs &args);
  FailureOr<Value> lowerMemFull(hc_front::CallOp call, StringRef method,
                                const CallArgs &args);

  // Consume a `hc_front.call` whose callee was classified as a layout
  // primitive (`ref.kind = "layout_op"`). Today the only such primitive
  // is `as_layout(value, descriptor)`. The descriptor argument arrives
  // as an SSA value defined by an `hc_front.name` whose own `ref` is
  // `kind = "layout"` carrying typed `#hc.expr` / DictAttr pieces; this
  // path reassembles them into a `LayoutAttr` and emits `hc.as_layout`.
  FailureOr<Value> lowerLayoutOpCall(hc_front::CallOp op, const RefInfo &ref);

  // Validate that an `as_layout(...)` call has exactly two positional
  // arguments and an optional `shape=` kwarg.
  LogicalResult validateAsLayoutCallShape(hc_front::CallOp op, ValueRange args);

  // `as_layout(value, descriptor, *, shape=?)` carries kwargs alongside
  // positionals on a single `hc_front.call` argument list. The descriptor
  // arrives as an `hc_front.name` whose `valueMap` entry is null by
  // design (its ref carries structured `LayoutAttr` pieces); separating
  // positionals from kwargs lets `lowerLayoutOpCall` lower each piece
  // through the right path.
  struct AsLayoutCallSplit {
    SmallVector<Value, 2> positional;
    Value shapeKwValue;
  };
  AsLayoutCallSplit splitAsLayoutCallArgs(ValueRange args);

  // Wrap a freshly-emitted tensor/vector result in `hc.as_layout` when
  // the originating call carried a `layout=` kwarg. Returns the
  Value tryLowerLaunchGeoCall(hc_front::CallOp call, StringRef method,
                              const CallArgs &args);

  unsigned getLaunchGeometryRank(const LaunchGeoMethodInfo &method,
                                 Type contextType,
                                 std::optional<unsigned> requiredRank) const;

  // Emit the launch-geometry op for a classified `group.{method}` DSL
  // attribute. Multi-axis queries return a single `hc.tuple` value wrapping
  // the full result vector; scalar queries return the scalar op result
  // directly.
  Value tryEmitLaunchGeo(const LaunchGeoMethodInfo &method, Value context,
                         Location loc,
                         std::optional<unsigned> requiredRank = std::nullopt);

  // Per-module counter used to mint unique prefixes for each
  // `hc_front.inlined_region` we flatten. Must strictly monotonically
  // increase across the full conversion so two call sites of the same
  // helper (even nested) never collide.
  unsigned inlineSiteCounter = 0;
};

//===----------------------------------------------------------------------===//
// Top-level callable dispatch.
//===----------------------------------------------------------------------===//

// Common skeleton for kernel/func lowering: build a fresh entry block with
// one block arg per param, hand (entry, fnType) to the per-kind `build`
// lambda which creates the hc op + attaches the block, then (if `build`
// asks for it) emit a leading `hc.assign` per param and lower the hc_front
// body region into that entry block. `build` returning a null `Region *`
// on success means "skip the walk".
//
// Block ownership: on `materializeParameters` failure we delete
// `entry` here; on success `build` is contractually attach-then-erase
// — attaches `entry` to the hc op first, so any later failure can
// `hcOp->erase()` and take the block with it.
LogicalResult Lowerer::runCallableBody(Operation *frontOp, ArrayAttr runParams,
                                       bool returnsValue, bool ensureReturn,
                                       BodyBuilder build) {
  Block *entry = new Block();
  auto fnType = materializeParameters(runParams, *entry, frontOp, returnsValue);
  if (failed(fnType)) {
    delete entry;
    return failure();
  }
  FailureOr<Region *> bodyRegion = build(entry, *fnType);
  if (failed(bodyRegion))
    return failure();
  if (!*bodyRegion)
    return success();
  OpBuilder::InsertionGuard bodyGuard(builder);
  emitParameterAssigns(runParams, *entry, frontOp->getLoc());
  if (failed(lowerRegion(**bodyRegion)))
    return failure();
  if (ensureReturn && (entry->empty() || !isa<HCReturnOp>(entry->back()))) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(entry);
    HCReturnOp::create(builder, frontOp->getLoc(), ValueRange{});
  }
  return success();
}

// Stamp shape/layout metadata onto a freshly built `hc.kernel`. Returns
// failure when a string-array `work_shape` / `group_shape` attribute can't
// be promoted to a typed `#hc.shape<...>`.
LogicalResult Lowerer::populateKernelMetadata(HCKernelOp hcKernel,
                                              Operation *frontOp,
                                              FunctionType fnType) {
  MLIRContext *ctx = frontOp->getContext();
  if (auto ws = frontOp->getAttrOfType<ArrayAttr>("work_shape")) {
    auto shape = stringArrayToShape(frontOp, ws);
    if (failed(shape)) {
      hcKernel->erase();
      return failure();
    }
    hcKernel.setWorkShapeAttr(*shape);
    workRank = static_cast<unsigned>((*shape).getDims().size());
  }
  if (auto gs = frontOp->getAttrOfType<ArrayAttr>("group_shape")) {
    auto shape = stringArrayToShape(frontOp, gs);
    if (failed(shape)) {
      hcKernel->erase();
      return failure();
    }
    hcKernel.setGroupShapeAttr(*shape);
    groupRank = static_cast<unsigned>((*shape).getDims().size());
  }
  if (auto sg = frontOp->getAttrOfType<IntegerAttr>("subgroup_size"))
    hcKernel.setSubgroupSizeAttr(sg);
  hcKernel.setBoundSymbolsAttr(buildKernelBoundSymbols(
      ctx, fnType.getInputs(), hcKernel.getWorkShapeAttr(),
      hcKernel.getGroupShapeAttr()));
  if (auto lits = frontOp->getAttrOfType<ArrayAttr>("literals"))
    hcKernel.setLiteralsAttr(lits);
  // `literal_bindings` rides on `hc_front.kernel` when the launcher
  // (`hc.compile(symbols={...})`) has pinned concrete integer values
  // for the specialization point. Carry it over so the downstream
  // `hc-specialize-literals` pass can fold them into the IR, and
  // augment it with the launch-geo `$`-prefixed bindings the
  // launcher can't see — `group_shape` / `work_shape` literal dims,
  // `subgroup_size`, static `group_size` — so symbolic launch-context
  // syms get folded the same way as user-named ones.
  DictionaryAttr launcherBindings =
      frontOp->getAttrOfType<DictionaryAttr>("literal_bindings");
  DictionaryAttr merged = augmentLiteralBindingsFromLaunchGeo(
      ctx, launcherBindings, hcKernel.getWorkShapeAttr(),
      hcKernel.getGroupShapeAttr(), hcKernel.getSubgroupSizeAttr());
  if (merged && !merged.empty())
    hcKernel.setLiteralBindingsAttr(merged);
  return success();
}

LogicalResult Lowerer::lowerKernelCallable(hc_front::KernelOp kernel,
                                           ArrayAttr params) {
  Operation *frontOp = kernel.getOperation();
  MLIRContext *ctx = frontOp->getContext();
  Location loc = frontOp->getLoc();
  return runCallableBody(
      frontOp, params, /*returnsValue=*/false, /*ensureReturn=*/false,
      [&](Block *entry, FunctionType fnType) -> FailureOr<Region *> {
        auto hcKernel = HCKernelOp::create(
            builder, loc, StringAttr::get(ctx, kernel.getName()),
            TypeAttr::get(fnType), /*work_shape=*/ShapeAttr(),
            /*group_shape=*/ShapeAttr(),
            /*subgroup_size=*/IntegerAttr(), /*bound_symbols=*/ArrayAttr(),
            /*literals=*/ArrayAttr(),
            /*literal_bindings=*/DictionaryAttr(),
            /*requirements=*/ConstraintSetAttr());
        hcKernel.getBody().push_back(entry);
        if (failed(populateKernelMetadata(hcKernel, frontOp, fnType)))
          return failure();
        return &kernel.getBody();
      });
}

LogicalResult Lowerer::lowerFuncCallable(hc_front::FuncOp func,
                                         ArrayAttr params) {
  Operation *frontOp = func.getOperation();
  MLIRContext *ctx = frontOp->getContext();
  Location loc = frontOp->getLoc();
  return runCallableBody(
      frontOp, params, /*returnsValue=*/!declaresNoneReturn(frontOp),
      /*ensureReturn=*/true,
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

// Validate the constraints between `keyword_only_parameters` and
// `const_kwargs`: the former must be a flat string list and every entry of
// the latter must appear in the former. The list itself isn't built here;
// it's recovered from `params` via `parameterNamesFromDicts` etc.
static LogicalResult
validateIntrinsicConstKwargs(Operation *frontOp, ArrayAttr constKwargsAttr,
                             ArrayAttr keywordOnlyParameters) {
  if (!constKwargsAttr)
    return success();
  for (auto [idx, kw] : llvm::enumerate(constKwargsAttr))
    if (!isa<StringAttr>(kw))
      return frontOp->emitOpError("`const_kwargs` entry at index ")
             << idx << " must be a StringAttr, got " << kw;
  llvm::SmallDenseSet<StringRef> keywordOnlySet;
  for (Attribute kw : keywordOnlyParameters)
    keywordOnlySet.insert(cast<StringAttr>(kw).getValue());
  for (Attribute kw : constKwargsAttr) {
    StringRef name = cast<StringAttr>(kw).getValue();
    if (!keywordOnlySet.contains(name))
      return frontOp->emitOpError("const kwarg '")
             << name << "' must be declared keyword-only";
  }
  return success();
}

// Build the runtime function-type for an intrinsic, applying the optional
// `operand_types` / `result_types` typed contract from the frontend.
FailureOr<FunctionType> Lowerer::buildIntrinsicFunctionType(
    Operation *frontOp, ArrayAttr parameterNames, ArrayAttr constKwargsAttr) {
  MLIRContext *ctx = frontOp->getContext();
  FunctionType fnType = getIntrinsicOperandFunctionType(
      parameterNames, constKwargsAttr, TypeRange{undef}, undef);
  if (auto operandTypesAttr =
          frontOp->getAttrOfType<ArrayAttr>("operand_types")) {
    FailureOr<SmallVector<Type>> operandTypes =
        typesFromContractArray(frontOp, operandTypesAttr, "operand_types");
    if (failed(operandTypes))
      return failure();
    if (operandTypes->size() != fnType.getNumInputs())
      return frontOp->emitOpError("`operand_types` declares ")
             << operandTypes->size()
             << " type(s) but intrinsic runtime signature has "
             << fnType.getNumInputs() << " SSA operand(s)";
    SmallVector<Type> resultTypes(fnType.getResults().begin(),
                                  fnType.getResults().end());
    fnType = FunctionType::get(ctx, *operandTypes, resultTypes);
  }
  if (auto resultTypesAttr =
          frontOp->getAttrOfType<ArrayAttr>("result_types")) {
    FailureOr<SmallVector<Type>> resultTypes =
        typesFromContractArray(frontOp, resultTypesAttr, "result_types");
    if (failed(resultTypes))
      return failure();
    fnType = FunctionType::get(ctx, fnType.getInputs(), *resultTypes);
  }
  return fnType;
}

LogicalResult Lowerer::lowerIntrinsicCallable(hc_front::IntrinsicOp intr,
                                              ArrayAttr params) {
  Operation *frontOp = intr.getOperation();
  MLIRContext *ctx = frontOp->getContext();
  Location loc = frontOp->getLoc();
  auto scopeAttr = frontOp->getAttrOfType<StringAttr>("scope");
  if (!scopeAttr)
    return frontOp->emitOpError(
        "hc_front.intrinsic must carry a `scope` string attribute");
  FailureOr<ArrayAttr> parameterNames =
      parameterNamesFromDicts(params, frontOp);
  if (failed(parameterNames))
    return failure();
  FailureOr<ArrayAttr> keywordOnlyParameters =
      keywordOnlyParametersFromDicts(params, frontOp);
  if (failed(keywordOnlyParameters))
    return failure();
  auto constKwargsAttr = frontOp->getAttrOfType<ArrayAttr>("const_kwargs");
  if (failed(validateIntrinsicConstKwargs(frontOp, constKwargsAttr,
                                          *keywordOnlyParameters)))
    return failure();

  // `hc.intrinsic` owns the const-kwarg filtering rule: its
  // `function_type` is the runtime operand signature, while
  // `parameters` keeps the full declared order for call-site kwarg
  // binding. Python metadata may publish a typed contract; otherwise
  // operands/results stay erased for target-specific validation.
  FailureOr<FunctionType> fnType =
      buildIntrinsicFunctionType(frontOp, *parameterNames, constKwargsAttr);
  if (failed(fnType))
    return failure();

  Block *entry = new Block();
  for (Type input : fnType->getInputs())
    entry->addArgument(input, loc);

  auto hcIntr = HCIntrinsicOp::create(
      builder, loc, StringAttr::get(ctx, intr.getName()),
      TypeAttr::get(*fnType), ScopeAttr::get(ctx, scopeAttr.getValue()),
      /*effects=*/EffectClassAttr(), /*const_kwargs=*/ArrayAttr(),
      /*parameters=*/*parameterNames,
      /*keyword_only=*/*keywordOnlyParameters);
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
  if (constKwargsAttr)
    hcIntr.setConstKwargsAttr(constKwargsAttr);
  return success();
}

LogicalResult Lowerer::lowerCallable(Operation *frontOp) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(frontOp);

  // The `parameters = [...]` attribute is populated by the Python driver
  // for every callable and carries the ordered param list (plus optional
  // structural annotations). Downstream we rely on both the arity and the
  // name -> block-arg binding, so a missing attr is programmer error.
  auto params = frontOp->getAttrOfType<ArrayAttr>("parameters");
  if (!params)
    return frontOp->emitOpError(
        "missing `parameters` attribute; the hc_front driver must stamp one");
  collectStaticLaunchGeometryRanks(frontOp);

  if (auto kernel = dyn_cast<hc_front::KernelOp>(frontOp))
    return lowerKernelCallable(kernel, params);
  if (auto func = dyn_cast<hc_front::FuncOp>(frontOp))
    return lowerFuncCallable(func, params);
  if (auto intr = dyn_cast<hc_front::IntrinsicOp>(frontOp))
    return lowerIntrinsicCallable(intr, params);
  return frontOp->emitOpError("unexpected top-level hc_front op");
}

// Scoped helpers (`scope=workitem` / `scope=subgroup`) require the first
// parameter that carries a launch-context kind to *match* the helper's
// declared scope. Walk to the first such entry and validate it; any other
// position diagnoses inside `validateLaunchContextParameter`.
LogicalResult Lowerer::validateScopedHelperLaunchContexts(Operation *sourceOp,
                                                          ArrayAttr params) {
  auto scopedExpected = getScopedLaunchContextParameterKind(sourceOp);
  if (!scopedExpected)
    return success();
  for (auto [idx, param] : llvm::enumerate(params)) {
    auto dict = dyn_cast<DictionaryAttr>(param);
    if (!dict)
      continue;
    std::optional<StringRef> launchContext =
        getLaunchContextParameterKind(dict);
    if (!launchContext)
      continue;
    return validateLaunchContextParameter(sourceOp, dict, idx, *launchContext,
                                          *scopedExpected);
  }
  return success();
}

// Capture launch geometry off whichever launch-context type just entered
// the parameter list. The three context types share the same trio of
// shape / size accessors but expose them under three distinct C++ types,
// so the dispatch is a small chain rather than a single template.
void Lowerer::recordLaunchContextShape(Type paramType) {
  if (auto groupType = dyn_cast<GroupType>(paramType)) {
    launchWorkShape = groupType.getWorkShape();
    launchGroupShape = groupType.getGroupShape();
    launchSubgroupSize = groupType.getSubgroupSize();
    if (launchWorkShape)
      workRank = static_cast<unsigned>(launchWorkShape.getDims().size());
    if (launchGroupShape)
      groupRank = static_cast<unsigned>(launchGroupShape.getDims().size());
    return;
  }
  if (auto workitemType = dyn_cast<WorkitemType>(paramType)) {
    launchGroupShape = workitemType.getGroupShape();
    launchSubgroupSize = workitemType.getSubgroupSize();
    if (launchGroupShape)
      groupRank = static_cast<unsigned>(launchGroupShape.getDims().size());
    return;
  }
  if (auto subgroupType = dyn_cast<SubgroupType>(paramType)) {
    launchGroupShape = subgroupType.getGroupShape();
    launchSubgroupSize = subgroupType.getSubgroupSize();
    if (launchGroupShape)
      groupRank = static_cast<unsigned>(launchGroupShape.getDims().size());
  }
}

FailureOr<FunctionType> Lowerer::materializeParameters(ArrayAttr params,
                                                       Block &entry,
                                                       Operation *sourceOp,
                                                       bool returnsValue) {
  MLIRContext *ctx = sourceOp->getContext();
  if (failed(validateScopedHelperLaunchContexts(sourceOp, params)))
    return failure();
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
    FailureOr<Type> paramType =
        parameterTypeFromDict(sourceOp, dict, undef, defaultLaunchMetadata,
                              static_cast<unsigned>(inputs.size()));
    if (failed(paramType))
      return failure();
    recordLaunchContextShape(*paramType);
    inputs.push_back(*paramType);
    entry.addArgument(*paramType, sourceOp->getLoc());
  }

  SmallVector<Type> results;
  if (returnsValue)
    results.push_back(undef);
  return FunctionType::get(ctx, inputs, results);
}

void Lowerer::emitParameterAssigns(ArrayAttr params, Block &entry,
                                   Location loc) {
  // Preconditions set up by `materializeParameters`: the dict shape of
  // every entry is already validated, and `entry` has exactly one block
  // argument per parameter. Assert the block-arg count here so a future caller
  // that forgets to call `materializeParameters` first trips a loud check
  // instead of an out-of-bounds read.
  assert(entry.getNumArguments() == params.size() &&
         "emitParameterAssigns: entry block arg count mismatch; "
         "caller must run materializeParameters first");
  // Drive our own insertion point: this helper is the sole producer of
  // parameter-entry `hc.assign` ops and the callers always want them at
  // the top of `entry`, so owning the position here removes a whole
  // class of subtle bug (wrong builder state after an intermediate
  // `RewriterBase::create`).
  builder.setInsertionPointToStart(&entry);
  for (auto [idx, param] : llvm::enumerate(params)) {
    auto dict = cast<DictionaryAttr>(param);
    auto name = cast<StringAttr>(dict.get("name"));
    HCAssignOp::create(builder, loc, name, entry.getArgument(idx));
  }
}

//===----------------------------------------------------------------------===//
// Per-region walk. Handles every `hc_front` op individually; unknown ops
// surface as a diagnostic rather than silent skip.
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerRegion(Region &src) {
  if (src.empty())
    return success();
  // Every `hc_front` region-carrying op declares its regions as
  // `SizedRegion<1>` (see HCFrontOps.td), so multi-block input is rejected
  // by the dialect verifier before this pass runs. We still keep a runtime
  // guard (not just `assert`) so a pipeline that bypasses verification —
  // fuzzing, hand-built IR, a buggy upstream transform — fails loud
  // instead of silently lowering only `front()` under NDEBUG.
  if (!src.hasOneBlock()) {
    Operation *parent = src.getParentOp();
    return (parent ? parent->emitOpError() : mlir::emitError(src.getLoc()))
           << "hc_front region must be single-block (dialect verifier was "
              "bypassed?)";
  }
  Block &block = src.front();
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (failed(lowerOp(&op)))
      return failure();
  }
  return success();
}

// Pure-syntax targets (LHS of an `hc_front.assign`): no `hc` op gets
// emitted, but operand lookups still need a map entry so a downstream
// pattern doesn't fault on the lookup.
static bool isLoweredAsTargetSyntax(Operation *op) {
  return isa<hc_front::TargetNameOp, hc_front::TargetTupleOp,
             hc_front::TargetSubscriptOp>(op);
}

// Build an `hc.tuple` from the lowered elements of `t`. A null lowered
// element is a hard error here: tuple elements are first-class SSA
// values and have no "consumed by parent" interpretation.
LogicalResult Lowerer::lowerTupleOp(hc_front::TupleOp t) {
  FailureOr<SmallVector<Value>> elts =
      lowerValueOperands(t.getElements(), t.getOperation(), "tuple element");
  if (failed(elts))
    return failure();
  if (llvm::any_of(*elts, [](Value v) { return !v; }))
    return t.emitOpError("tuple element did not lower to an hc value");
  SmallVector<Type> elementTypes;
  elementTypes.reserve(elts->size());
  for (Value elt : *elts)
    elementTypes.push_back(elt.getType());
  Type tupleType = TupleType::get(t.getContext(), elementTypes);
  valueMap[t.getResult()] =
      HCTupleOp::create(builder, t.getLoc(), tupleType, *elts);
  return success();
}

// `keyword(name, value)` is a passthrough for the lowered value paired
// with its name; it isn't itself an SSA-bearing op on the `hc` side.
LogicalResult Lowerer::lowerKeywordOp(hc_front::KeywordOp k) {
  FailureOr<Value> v =
      lowerValueOperand(k.getValue(), k.getOperation(), "keyword value");
  if (failed(v))
    return failure();
  keywordInfo[k.getResult()] = {k.getName(), *v};
  valueMap[k.getResult()] = Value();
  return success();
}

// Bucket of producing ops whose lowering returns a `FailureOr<Value>` and
// where a successful null lowering is meaningful (the parent op consumes
// the result directly): name/attr/call.
LogicalResult Lowerer::lowerProducingOp(Operation *op) {
  auto record = [&](Value result, FailureOr<Value> lowered) {
    if (failed(lowered))
      return failure();
    valueMap[result] = *lowered;
    return success();
  };
  if (auto name = dyn_cast<hc_front::NameOp>(op))
    return record(name.getResult(), lowerName(name));
  if (auto attr = dyn_cast<hc_front::AttrOp>(op))
    return record(attr.getResult(), lowerAttr(attr));
  if (auto c = dyn_cast<hc_front::CallOp>(op))
    return record(c.getResult(), lowerCall(c));
  return failure();
}

// Bucket of value-or-null producers: a null lowering is a hard failure
// (no "consumed by parent" sentinel for these).
LogicalResult Lowerer::lowerScalarValueOp(Operation *op) {
  auto record = [&](Value result, Value v) {
    valueMap[result] = v;
    return v ? success() : failure();
  };
  if (auto c = dyn_cast<hc_front::ConstantOp>(op)) {
    valueMap[c.getResult()] = lowerConstant(c);
    return success();
  }
  if (auto b = dyn_cast<hc_front::BinOp>(op))
    return record(b.getResult(), lowerBinop(b));
  if (auto s = dyn_cast<hc_front::SliceOp>(op))
    return record(s.getResult(), lowerSlice(s));
  if (auto s = dyn_cast<hc_front::SubscriptOp>(op))
    return record(s.getResult(), lowerSubscript(s));
  return failure();
}

// Bucket of structural / control-flow ops whose lowering already returns
// `LogicalResult` directly.
LogicalResult Lowerer::lowerStructuralOp(Operation *op) {
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
  if (auto ir = dyn_cast<hc_front::InlinedRegionOp>(op))
    return lowerInlinedRegion(ir);
  if (auto t = dyn_cast<hc_front::TupleOp>(op))
    return lowerTupleOp(t);
  if (auto k = dyn_cast<hc_front::KeywordOp>(op))
    return lowerKeywordOp(k);
  return failure();
}

LogicalResult Lowerer::lowerOp(Operation *op) {
  if (isLoweredAsTargetSyntax(op)) {
    valueMap[op->getResult(0)] = Value();
    return success();
  }
  if (isa<hc_front::NameOp, hc_front::AttrOp, hc_front::CallOp>(op))
    return lowerProducingOp(op);
  if (isa<hc_front::ConstantOp, hc_front::BinOp, hc_front::SliceOp,
          hc_front::SubscriptOp>(op))
    return lowerScalarValueOp(op);
  if (isa<hc_front::ReturnOp, hc_front::AssignOp, hc_front::ForOp,
          hc_front::WorkitemRegionOp, hc_front::SubgroupRegionOp,
          hc_front::InlinedRegionOp, hc_front::TupleOp, hc_front::KeywordOp>(
          op))
    return lowerStructuralOp(op);
  return op->emitOpError("unsupported hc_front op");
}

// Returns the lowered hc value for an hc_front SSA operand. Null is a
// deliberate sentinel meaning "no SSA counterpart" (keywords, callee-like
// names, attr chains) — new consumers must either treat it as
// consumed-by-parent or null-check and diagnose; binding it into a scope or
// handing it to an hc op builder silently produces bad IR.
FailureOr<Value> Lowerer::lowerValueOperand(Value v, Operation *consumer,
                                            StringRef role) {
  auto it = valueMap.find(v);
  if (it == valueMap.end()) {
    consumer->emitOpError(role)
        << " was not lowered before use; producer appears later or outside "
           "the converted region";
    return failure();
  }
  return it->second;
}

FailureOr<SmallVector<Value>> Lowerer::lowerValueOperands(ValueRange vs,
                                                          Operation *consumer,
                                                          StringRef role) {
  SmallVector<Value> out;
  out.reserve(vs.size());
  for (Value v : vs) {
    FailureOr<Value> lowered = lowerValueOperand(v, consumer, role);
    if (failed(lowered))
      return failure();
    out.push_back(*lowered);
  }
  return out;
}

FailureOr<SmallVector<Value>>
Lowerer::expandTupleOperand(Value v, Operation *consumer, StringRef role) {
  FailureOr<Value> lowered = lowerValueOperand(v, consumer, role);
  if (failed(lowered))
    return failure();
  if (!*lowered)
    return failure();
  // Only syntax sites that are genuinely variadic call this helper. Everywhere
  // else an `hc.tuple` remains one first-class SSA value.
  if (auto tuple = (*lowered).getDefiningOp<HCTupleOp>()) {
    SmallVector<Value> elements;
    elements.append(tuple.getElements().begin(), tuple.getElements().end());
    return elements;
  }
  return SmallVector<Value>{*lowered};
}

FailureOr<SmallVector<Value>>
Lowerer::lowerReturnValues(hc_front::ReturnOp op) {
  SmallVector<Value> values;
  values.reserve(op.getValues().size());
  for (Value value : op.getValues()) {
    FailureOr<Value> lowered =
        lowerValueOperand(value, op.getOperation(), "return operand");
    if (failed(lowered))
      return failure();
    if (!*lowered)
      return op.emitOpError("return operand did not lower");
    values.push_back(*lowered);
  }
  return values;
}

//===----------------------------------------------------------------------===//
// Name / attr. The classification on `ref` drives everything.
//===----------------------------------------------------------------------===//

FailureOr<Value> Lowerer::lowerName(hc_front::NameOp op) {
  RefInfo ref = RefInfo::get(op);
  if (failed(ref.diagnoseIfMalformed(op)))
    return failure();
  StringRef kind = ref.getKind();
  StringRef ident = op.getName();
  if (kind == "param" || kind == "local" || kind == "iv") {
    // `hc.name_load` is a placeholder the promotion pass folds into the
    // reaching SSA definition. An unresolved read (no reaching
    // `hc.assign`) surfaces a diagnostic there, not here — this pass
    // does not know what names will be bound by siblings / ancestors
    // yet to be walked.
    auto name = StringAttr::get(op.getContext(), ident);
    return HCNameLoadOp::create(builder, op.getLoc(), undef, name).getResult();
  }
  if (kind == "constant")
    return emitConstantFromRef(op.getLoc(), ref, op);
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

// Closest ancestor (inclusive of self's parent chain) carrying a
// front-pass `parameters` attribute — typically the enclosing kernel
// or helper-function op. Lets the alias dispatch below stay agnostic
// to which front-pass region kind owns the parameters list.
static Operation *findFrontParamsHost(Operation *startOp) {
  for (Operation *parent = startOp->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->hasAttr("parameters"))
      return parent;
  return nullptr;
}

// Linear scan of the `parameters` array for the entry whose `name`
// string matches. Returns a null `DictionaryAttr` on miss.
static DictionaryAttr findFrontParamDict(Operation *paramHost, StringRef name) {
  auto params = paramHost->getAttrOfType<ArrayAttr>("parameters");
  if (!params)
    return {};
  for (Attribute paramAttr : params) {
    auto dict = dyn_cast<DictionaryAttr>(paramAttr);
    if (!dict)
      continue;
    auto nameAttr = dict.getAs<StringAttr>("name");
    if (nameAttr && nameAttr.getValue() == name)
      return dict;
  }
  return {};
}

// Walks the enclosing front-pass region(s) for a `parameters` dict
// and decides whether `nameOp` binds to a launch-context entry. The
// lowered base value's MLIR type is the erased `!hc.undef` placeholder
// at conversion time — the kernel block argument only acquires its
// launch-context (`!hc.group` / `!hc.workitem` / `!hc.subgroup`) type
// after `-hc-promote-names` — so the front-pass `parameters` attribute
// is the only authoritative source available here.
//
// Used to disambiguate the Python-side `shape` alias (which on
// `CurrentGroup` / `WorkItem` / `SubGroup` is the launch-geo
// `group_shape` getter) from buffer / tensor shape queries on the
// same attribute name.
static bool isLaunchContextFrontParam(hc_front::NameOp nameOp) {
  RefInfo ref = RefInfo::get(nameOp);
  if (ref.getKind() != "param")
    return false;
  StringRef name = nameOp.getName();
  Operation *paramHost = findFrontParamsHost(nameOp.getOperation());
  if (!paramHost)
    return false;
  DictionaryAttr param = findFrontParamDict(paramHost, name);
  if (!param)
    return false;
  if (auto kindAttr = param.getAs<StringAttr>("kind"))
    if (kindAttr.getValue() == "launch_context")
      return true;
  // `buildImplicitGroupParameterType` synthesizes a `!hc.group` type
  // for a non-scoped kernel parameter literally named `group` even
  // when no explicit `launch_context` kind is stamped; mirror that
  // here so the alias dispatch agrees with parameter typing.
  bool scoped = !!paramHost->getAttrOfType<StringAttr>("scope");
  return !scoped && name == "group";
}

static bool isLaunchContextFrontBase(Value baseFront) {
  auto nameOp =
      dyn_cast_if_present<hc_front::NameOp>(baseFront.getDefiningOp());
  return nameOp && isLaunchContextFrontParam(nameOp);
}

FailureOr<Value> Lowerer::lowerAttr(hc_front::AttrOp op) {
  // Most attribute chains (`group.load`, `buf.shape`, ...) are folded
  // into the parent call or subscript and don't produce a standalone
  // SSA value at this layer. Two exceptions materialize eagerly here:
  //
  //  * `numpy_dtype_type` denotes a type. Both `x.astype(np.<dtype>)`
  //    (arg position) and `np.<dtype>(0)` (callee position) need to
  //    observe an `hc.const` carrying a `TypeAttr`. Emitting it here
  //    keeps `collectCallArgs` honest (no null-arg rejection) and lets
  //    `lowerDslMethodCall` reuse the same value when a numpy dtype is
  //    used as a value constructor.
  //
  //  * Launch-geometry getters (`group.work_offset`, `wi.local_id`,
  //    ...). The Python frontend can bind these to locals
  //    (`gid = group.work_offset`) or thread them through helpers;
  //    leaving the bare attr as a null Value used to crash assignment
  //    lowering with "rhs did not lower to an hc value". Lowering
  //    eagerly to the multi-axis tuple (or scalar) makes the value a
  //    first-class SSA carrier, and the inline-subscript /
  //    call-of-getter folds just retrieve it from the cache instead of
  //    re-emitting their own launch-geo op.
  //
  // A present-but-malformed `ref` (dict without a string `kind`) is
  // diagnosed here so the error fingers the attr rather than the
  // downstream call/subscript that was trying to read `method`.
  RefInfo ref = RefInfo::get(op);
  if (failed(ref.diagnoseIfMalformed(op)))
    return failure();
  if (ref.getKind() == "numpy_dtype_type") {
    StringRef dtype = ref.getString("dtype");
    std::optional<Type> dtypeTy = resolveNumpyDtypeType(op.getContext(), dtype);
    if (!dtypeTy) {
      op.emitOpError("unsupported numpy_dtype_type '") << dtype << "'";
      return failure();
    }
    return {
        HCConstOp::create(builder, op.getLoc(), undef, TypeAttr::get(*dtypeTy))
            .getResult()};
  }
  // Strict (name-only) classification handles the canonical method
  // spellings — `group_id`, `local_id`, `work_offset`, `group_shape`,
  // etc. The Python-side `CurrentGroup.shape` alias collides by name
  // with buffer / tensor `.shape`, so disambiguate by walking the
  // front-pass `parameters` dict on the enclosing kernel — that's the
  // earliest checkpoint where the base's launch-context-ness is
  // available (lowered base types are still the erased placeholder).
  std::optional<LaunchGeoMethodInfo> info =
      classifyLaunchGeoMethod(op.getName());
  if (!info && op.getName() == "shape" &&
      isLaunchContextFrontBase(op.getBase()))
    info = getLaunchGeoMethodInfo(LaunchGeoMethod::GroupShape);
  if (!info)
    return Value();
  FailureOr<Value> baseOr =
      lowerValueOperand(op.getBase(), op.getOperation(), "launch-geo base");
  if (failed(baseOr))
    return failure();
  Value base = *baseOr;
  if (!base)
    return Value();
  std::optional<unsigned> requiredRank =
      getStaticLaunchGeometryRank(op.getBase(), op.getName());
  return tryEmitLaunchGeo(*info, base, op.getLoc(), requiredRank);
}

FailureOr<Value> Lowerer::emitConstantFromRef(Location loc, const RefInfo &ref,
                                              Operation *sourceOp) {
  MLIRContext *ctx = sourceOp->getContext();
  StringRef pyKind = ref.getString("python_kind");
  StringRef raw = ref.getString("value");
  Attribute payload;
  if (pyKind == "int") {
    int64_t v = 0;
    if (raw.getAsInteger(10, v)) {
      return sourceOp->emitOpError("constant value '")
             << raw << "' not parseable as int";
    }
    payload = IntegerAttr::get(IntegerType::get(ctx, 64), v);
  } else if (pyKind == "float") {
    APFloat f(APFloat::IEEEdouble());
    auto status = f.convertFromString(raw, APFloat::rmNearestTiesToEven);
    if (!status) {
      llvm::consumeError(status.takeError());
      return sourceOp->emitOpError("constant value '")
             << raw << "' not parseable as float";
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
    return sourceOp->emitOpError("unsupported constant python_kind '")
           << pyKind << "'";
  }
  return HCConstOp::create(builder, loc, undef, payload).getResult();
}

//===----------------------------------------------------------------------===//
// Simple value producers.
//===----------------------------------------------------------------------===//

Value Lowerer::lowerConstant(hc_front::ConstantOp op) {
  return HCConstOp::create(builder, op.getLoc(), undef, op.getValue());
}

Value Lowerer::lowerBinop(hc_front::BinOp op) {
  FailureOr<Value> lhsOr =
      lowerValueOperand(op.getLhs(), op.getOperation(), "lhs");
  FailureOr<Value> rhsOr =
      lowerValueOperand(op.getRhs(), op.getOperation(), "rhs");
  if (failed(lhsOr) || failed(rhsOr))
    return nullptr;
  Value lhs = *lhsOr;
  Value rhs = *rhsOr;
  if (!lhs || !rhs) {
    StringRef which =
        !lhs && !rhs ? StringRef("lhs+rhs") : (!lhs ? "lhs" : "rhs");
    op.emitOpError("binop operand did not lower to an hc value (")
        << which << "); operand may be a callee-like ref or parent-consumed "
        << "syntax node";
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

  FailureOr<SmallVector<Value>> partsOr =
      lowerValueOperands(op.getParts(), op.getOperation(), "slice operand");
  if (failed(partsOr))
    return nullptr;
  SmallVector<Value> parts = std::move(*partsOr);
  if (llvm::any_of(parts, [](Value v) { return !v; })) {
    op.emitOpError("slice operand did not lower to an hc value");
    return nullptr;
  }
  size_t expected = unsigned(hasLower) + unsigned(hasUpper) + unsigned(hasStep);
  if (parts.size() != expected) {
    op.emitOpError("slice operand count ")
        << parts.size()
        << " does not match has_lower/has_upper/has_step flags "
           "(expected "
        << expected << ")";
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

  return HCSliceExprOp::create(builder, op.getLoc(), undef, lo, hi, st);
}

//===----------------------------------------------------------------------===//
// Control flow / binding.
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerReturn(hc_front::ReturnOp op) {
  FailureOr<SmallVector<Value>> values = lowerReturnValues(op);
  if (failed(values))
    return failure();
  HCReturnOp::create(builder, op.getLoc(), *values);
  return success();
}

LogicalResult Lowerer::lowerAssignToName(hc_front::AssignOp op,
                                         hc_front::TargetNameOp tn) {
  FailureOr<Value> valueOr =
      lowerValueOperand(op.getValue(), op.getOperation(), "assignment rhs");
  if (failed(valueOr))
    return failure();
  Value value = *valueOr;
  if (!value)
    return op.emitOpError("rhs for '") << tn.getName()
                                       << "' did not lower to an hc value; "
                                          "ref classification may be off";
  HCAssignOp::create(builder, op.getLoc(),
                     StringAttr::get(op.getContext(), tn.getName()), value);
  return success();
}

// Multi-assign: `a, b = <rhs>`. First-class tuple SSA values are destructured
// with `hc.getitem`; truly multi-result front ops can still distribute their
// individual results. Arity-1 unpack also uses getitem so `a, = scalar` does
// not silently become scalar assignment.
FailureOr<SmallVector<Value>>
Lowerer::expandTupleAssignSources(hc_front::AssignOp op, Value value,
                                  size_t arity) {
  SmallVector<Value> sources;
  Operation *rhsOp = value ? value.getDefiningOp() : nullptr;
  if (rhsOp && rhsOp->getNumResults() == arity && arity != 1) {
    sources.assign(rhsOp->getResults().begin(), rhsOp->getResults().end());
    return sources;
  }
  if (!value)
    return op.emitOpError("tuple-unpack rhs did not lower to an hc value");
  MLIRContext *ctx = op.getContext();
  for (size_t i = 0; i < arity; ++i) {
    auto index = HCConstOp::create(
        builder, op.getLoc(), undef,
        IntegerAttr::get(IntegerType::get(ctx, 64), static_cast<int64_t>(i)));
    sources.push_back(HCGetItemOp::create(builder, op.getLoc(), undef, value,
                                          index.getResult()));
  }
  return sources;
}

LogicalResult Lowerer::lowerAssignToTuple(hc_front::AssignOp op,
                                          hc_front::TargetTupleOp tt) {
  size_t arity = tt.getElements().size();
  FailureOr<Value> valueOr =
      lowerValueOperand(op.getValue(), op.getOperation(), "tuple-unpack rhs");
  if (failed(valueOr))
    return failure();
  FailureOr<SmallVector<Value>> sourcesOr =
      expandTupleAssignSources(op, *valueOr, arity);
  if (failed(sourcesOr))
    return failure();
  if (sourcesOr->size() != arity)
    return op.emitOpError("tuple-unpack arity mismatch: rhs has ")
           << sourcesOr->size() << ", target has " << arity;
  MLIRContext *ctx = op.getContext();
  for (auto [elem, src] : llvm::zip(tt.getElements(), *sourcesOr)) {
    auto tn = dyn_cast_if_present<hc_front::TargetNameOp>(elem.getDefiningOp());
    if (!tn)
      return op.emitOpError("nested target kinds are not yet supported");
    if (!src)
      return op.emitOpError(
          "tuple-unpack source element did not lower to an hc value");
    HCAssignOp::create(builder, op.getLoc(), StringAttr::get(ctx, tn.getName()),
                       src);
  }
  return success();
}

LogicalResult Lowerer::lowerAssignToSubscript(hc_front::AssignOp op,
                                              hc_front::TargetSubscriptOp ts) {
  FailureOr<Value> valueOr =
      lowerValueOperand(op.getValue(), op.getOperation(), "store source");
  if (failed(valueOr))
    return failure();
  Value value = *valueOr;
  if (!value)
    return op.emitOpError("store source did not lower to an hc value");
  FailureOr<Value> baseOr = lowerValueOperand(ts.getBase(), op.getOperation(),
                                              "target_subscript base");
  if (failed(baseOr))
    return failure();
  Value base = *baseOr;
  if (!base)
    return op.emitOpError("target_subscript base is unresolved");
  SmallVector<Value> indices;
  for (Value idx : ts.getIndices()) {
    FailureOr<SmallVector<Value>> expanded =
        expandTupleOperand(idx, op.getOperation(), "target_subscript index");
    if (failed(expanded))
      return op.emitOpError("target_subscript index did not lower");
    indices.append(expanded->begin(), expanded->end());
  }
  HCStoreOp::create(builder, op.getLoc(), base, indices, value, Value{});
  return success();
}

LogicalResult Lowerer::lowerAssign(hc_front::AssignOp op) {
  Operation *target = op.getTarget().getDefiningOp();
  if (auto tn = dyn_cast_if_present<hc_front::TargetNameOp>(target))
    return lowerAssignToName(op, tn);
  if (auto tt = dyn_cast_if_present<hc_front::TargetTupleOp>(target))
    return lowerAssignToTuple(op, tt);
  if (auto ts = dyn_cast_if_present<hc_front::TargetSubscriptOp>(target))
    return lowerAssignToSubscript(op, ts);
  return op.emitOpError("unsupported assign target");
}

// Lower the `iter` region of `for` (`hc_front.for_range`'s iterator clause)
// in-place into the enclosing block and return the trailing call op, which
// must be the `range(...)` builtin.
FailureOr<hc_front::CallOp> Lowerer::lowerForIterRegion(hc_front::ForOp op,
                                                        Region &iter) {
  if (iter.empty() || iter.front().empty())
    return op.emitOpError("for-iter region is empty");
  // `hc_front.for` declares all three sub-regions as `SizedRegion<1>`, so
  // multi-block bodies cannot reach this pass — but keep a runtime check
  // so a verifier-bypass doesn't silently walk only the first block under
  // NDEBUG.
  if (!iter.hasOneBlock())
    return op.emitOpError(
        "for-iter must be single-block (dialect verifier bypassed?)");
  Operation *lastOp = nullptr;
  for (Operation &child : llvm::make_early_inc_range(iter.front())) {
    if (failed(lowerOp(&child)))
      return failure();
    lastOp = &child;
  }
  auto iterCall = dyn_cast_if_present<hc_front::CallOp>(lastOp);
  if (!iterCall)
    return op.emitOpError("for-iter must end in a call op");
  auto calleeName = dyn_cast_if_present<hc_front::NameOp>(
      iterCall.getCallee().getDefiningOp());
  RefInfo calleeRef = RefInfo::get(calleeName);
  if (!calleeName || calleeRef.getKind() != "builtin" ||
      calleeRef.getString("builtin") != "range")
    return op.emitOpError("for-iter must be `range(...)`");
  return iterCall;
}

// Python-style `range(stop)` / `range(start, stop)` / `range(start, stop,
// step)`: pad the missing parts with the canonical defaults so the
// `hc.for_range` op always sees three operands.
FailureOr<std::array<Value, 3>>
Lowerer::lowerForRangeArgs(hc_front::ForOp op, hc_front::CallOp iterCall) {
  FailureOr<SmallVector<Value>> rangeArgsOr = lowerValueOperands(
      iterCall.getArguments(), op.getOperation(), "range argument");
  if (failed(rangeArgsOr))
    return failure();
  SmallVector<Value> rangeArgs = std::move(*rangeArgsOr);
  if (llvm::any_of(rangeArgs, [](Value v) { return !v; }))
    return iterCall.emitOpError("range argument did not lower to an hc value");
  auto i64Const = [&](int64_t v) -> Value {
    return HCConstOp::create(
        builder, op.getLoc(), undef,
        IntegerAttr::get(IntegerType::get(op.getContext(), 64), v));
  };
  if (rangeArgs.size() == 1)
    return std::array<Value, 3>{i64Const(0), rangeArgs[0], i64Const(1)};
  if (rangeArgs.size() == 2)
    return std::array<Value, 3>{rangeArgs[0], rangeArgs[1], i64Const(1)};
  if (rangeArgs.size() == 3)
    return std::array<Value, 3>{rangeArgs[0], rangeArgs[1], rangeArgs[2]};
  return op.emitOpError("range(...) must have 1, 2, or 3 args");
}

LogicalResult Lowerer::lowerFor(hc_front::ForOp op) {
  FailureOr<hc_front::CallOp> iterCallOr = lowerForIterRegion(op, op.getIter());
  if (failed(iterCallOr))
    return failure();
  FailureOr<std::array<Value, 3>> rangeOr = lowerForRangeArgs(op, *iterCallOr);
  if (failed(rangeOr))
    return failure();
  Value lo = (*rangeOr)[0], hi = (*rangeOr)[1], step = (*rangeOr)[2];

  // Pull the IV name out of the target region. Same SizedRegion<1>
  // guarantee as above.
  Region &tgt = op.getTarget();
  if (tgt.empty() || tgt.front().empty())
    return op.emitOpError("for-target region is empty");
  if (!tgt.hasOneBlock())
    return op.emitOpError(
        "for-target must be single-block (dialect verifier bypassed?)");
  auto ivTarget = dyn_cast<hc_front::TargetNameOp>(&tgt.front().front());
  if (!ivTarget)
    return op.emitOpError("for-target must be a single target_name");

  // Build the `hc.for_range` with no iter_args — loop-carried value
  // analysis is a later pass. The `hc.assign "<iv>", %iv` emitted as
  // the first body op is the IV self-bind placeholder documented on
  // `hc.assign` in HCOps.td; promotion matches and folds it into
  // direct uses of the block arg.
  auto forOp = HCForRangeOp::create(builder, op.getLoc(),
                                    /*resultTypes=*/TypeRange{}, lo, hi, step,
                                    /*iter_inits=*/ValueRange{});
  Block *body = new Block();
  BlockArgument iv = body->addArgument(undef, op.getLoc());
  forOp.getBody().push_back(body);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  HCAssignOp::create(builder, op.getLoc(),
                     StringAttr::get(op.getContext(), ivTarget.getName()), iv);
  LogicalResult r = lowerRegion(op.getBody());
  if (succeeded(r))
    HCYieldOp::create(builder, op.getLoc());
  return r;
}

// Alpha-rename every local-identifier-bearing op in `body` by prefixing
// its name attribute with `prefix`. Scope: `hc_front.name` ops whose
// `ref.kind` is a local binding (`param`/`local`/`iv`) and every
// `hc_front.target_name`. Attribute names on `hc_front.attr`,
// `hc_front.keyword` etc. are Python syntactic tokens rather than
// bindings, so they are deliberately untouched.
//
// Walks nested regions inside the same lexical scope (e.g.
// `hc_front.for` / `hc_front.workitem_region`), but stops at any
// nested `hc_front.inlined_region` — that one is its own name
// boundary and gets rewritten with its own prefix when it in turn is
// flattened. Touching a nested inlined body here would double-prefix
// its params and break the `parameters` -> body mapping.
// True iff the `kind` field of an inline-rename `ref` dict is one of
// the alpha-renameable categories (function parameters, locals, IVs).
// Anything else (frontend names referencing module-level callables,
// captured constants, etc.) keeps its identity across inlining.
static bool isInlineRenameableNameKind(StringRef k) {
  return k == "param" || k == "local" || k == "iv";
}

// Rename a single `hc_front.name` op when its `ref` kind is one of
// the alpha-renameable categories.
static void alphaRenameNameOp(hc_front::NameOp name, StringRef prefix) {
  auto ref = name->getAttrOfType<DictionaryAttr>("ref");
  if (!ref)
    return;
  auto kind = ref.getAs<StringAttr>("kind");
  if (!kind || !isInlineRenameableNameKind(kind.getValue()))
    return;
  name.setName((prefix + name.getName()).str());
}

static void alphaRenameInlinedBody(Region &body, StringRef prefix) {
  SmallVector<Region *> worklist{&body};
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    for (Block &block : *region)
      for (Operation &op : block) {
        if (auto name = dyn_cast<hc_front::NameOp>(&op)) {
          alphaRenameNameOp(name, prefix);
          continue;
        }
        if (auto target = dyn_cast<hc_front::TargetNameOp>(&op)) {
          target.setName((prefix + target.getName()).str());
          continue;
        }
        // Respect nested inline name boundaries — their bodies get
        // their own per-site prefix when flattened.
        if (isa<hc_front::InlinedRegionOp>(&op))
          continue;
        for (Region &nested : op.getRegions())
          worklist.push_back(&nested);
      }
  }
}

// Bind each parameter of an inlined region to its caller-side argument
// via `hc.assign @<renamed_param> = <lowered_arg>`. The body's
// `hc_front.name` references to the param now read the prefixed name,
// which the promotion pass folds against these assigns.
LogicalResult Lowerer::bindInlinedRegionParams(hc_front::InlinedRegionOp op,
                                               ArrayAttr params,
                                               StringRef prefix) {
  MLIRContext *ctx = op.getContext();
  for (auto [paramAttr, arg] : llvm::zip_equal(params, op.getArguments())) {
    auto dict = dyn_cast<DictionaryAttr>(paramAttr);
    if (!dict)
      return op.emitOpError("invalid parameters entry");
    auto nameAttr = dict.getAs<StringAttr>("name");
    if (!nameAttr)
      return op.emitOpError("parameters entry missing `name`");
    FailureOr<Value> loweredArgOr =
        lowerValueOperand(arg, op.getOperation(), "inline argument");
    if (failed(loweredArgOr))
      return failure();
    Value loweredArg = *loweredArgOr;
    if (!loweredArg)
      return op.emitOpError("inline argument for param `")
             << nameAttr.getValue() << "' did not lower to an hc value";
    std::string renamed = (prefix + nameAttr.getValue()).str();
    HCAssignOp::create(builder, op.getLoc(), StringAttr::get(ctx, renamed),
                       loweredArg);
  }
  return success();
}

// Walk the inlined body, lowering every op via the standard dispatch
// except `hc_front.return`: that op's operands are the *region results*,
// not a real `hc.return` to the enclosing function. Tuple returns are
// first-class values; only explicit multi-operand returns create multiple
// region results.
FailureOr<SmallVector<Value>>
Lowerer::lowerInlinedRegionBody(hc_front::InlinedRegionOp op, Block &block) {
  SmallVector<Value> resultValues;
  bool sawReturn = false;
  for (Operation &child : llvm::make_early_inc_range(block)) {
    if (auto r = dyn_cast<hc_front::ReturnOp>(&child)) {
      if (sawReturn)
        return op.emitOpError("inlined_region body has multiple returns");
      sawReturn = true;
      FailureOr<SmallVector<Value>> loweredOr = lowerReturnValues(r);
      if (failed(loweredOr))
        return failure();
      resultValues = std::move(*loweredOr);
      continue;
    }
    if (failed(lowerOp(&child)))
      return failure();
  }
  if (!sawReturn)
    return op.emitOpError("inlined_region body has no return");
  return resultValues;
}

// Map the inlined region's results into `valueMap`. `hc_front.call` has
// one SSA result, so the canonical lowering for multiple explicit return
// operands is one tuple value at result #0; additional region results
// keep hand-written direct uses well-defined, with arity pinned by the
// front dialect verifier.
LogicalResult
Lowerer::publishInlinedRegionResults(hc_front::InlinedRegionOp op,
                                     ArrayRef<Value> resultValues) {
  unsigned nResults = op.getNumResults();
  if (nResults == 0) {
    if (!resultValues.empty())
      return op.emitOpError(
          "inlined_region declares no results but body returns values");
    return success();
  }
  if (resultValues.size() != nResults)
    return op.emitOpError(
               "inlined_region result arity mismatch: region declares ")
           << nResults << ", body returns " << resultValues.size();
  if (nResults == 1) {
    valueMap[op.getResult(0)] = resultValues.front();
    return success();
  }
  SmallVector<Type> resultTypes;
  resultTypes.reserve(resultValues.size());
  for (Value result : resultValues)
    resultTypes.push_back(result.getType());
  Type tupleType = TupleType::get(op.getContext(), resultTypes);
  valueMap[op.getResult(0)] =
      HCTupleOp::create(builder, op.getLoc(), tupleType, resultValues);
  for (unsigned i = 1; i < nResults; ++i)
    valueMap[op.getResult(i)] = resultValues[i];
  return success();
}

LogicalResult Lowerer::lowerInlinedRegion(hc_front::InlinedRegionOp op) {
  Region &body = op.getBody();
  if (body.empty() || !body.hasOneBlock())
    return op.emitOpError("inlined_region body must be single-block");

  auto params = op->getAttrOfType<ArrayAttr>("parameters");
  if (!params || params.size() != op.getArguments().size())
    return op.emitOpError(
               "inlined_region parameter/argument arity mismatch (params=")
           << (params ? params.size() : 0)
           << ", args=" << op.getArguments().size() << ")";

  // Per-site prefix. `inlineSiteCounter` is module-wide, so two sites
  // of the same helper in the same caller still get distinct prefixes.
  std::string prefix =
      ("__inl_" + op.getCallee() + "_" + Twine(inlineSiteCounter) + "_").str();
  ++inlineSiteCounter;

  alphaRenameInlinedBody(body, prefix);
  if (failed(bindInlinedRegionParams(op, params, prefix)))
    return failure();
  FailureOr<SmallVector<Value>> resultValuesOr =
      lowerInlinedRegionBody(op, body.front());
  if (failed(resultValuesOr))
    return failure();
  return publishInlinedRegionResults(op, *resultValuesOr);
}

// Tail-return regions (`return inner()` already folded into a region with
// `tail_return` set) must be the trailing op in their block and carry a
// single return; the result arity is the return's value arity.
template <typename FrontRegionOpT>
static FailureOr<unsigned> captureRegionTailReturnArity(FrontRegionOpT op) {
  if (op.getBody().empty())
    return op.emitOpError("tail-return region has no body");
  if (std::next(Block::iterator(op.getOperation())) != op->getBlock()->end())
    return op.emitOpError("tail-return region must be the final operation "
                          "in its enclosing block");
  hc_front::ReturnOp soleReturn;
  for (Operation &child : op.getBody().front()) {
    auto candidate = dyn_cast<hc_front::ReturnOp>(&child);
    if (!candidate)
      continue;
    if (soleReturn)
      return op.emitOpError("tail-return region has multiple returns");
    soleReturn = candidate;
  }
  if (!soleReturn)
    return op.emitOpError("tail-return region has no return");
  return static_cast<unsigned>(soleReturn.getValues().size());
}

// Materialize block arguments for the captured region's parameters and
// collect their published names so we can emit `hc.assign "<p>", %arg` at
// body entry. The first parameter must always carry an explicit launch-
// context marker matching the region kind.
template <typename FrontRegionOpT>
LogicalResult Lowerer::populateCapturingRegionParams(
    FrontRegionOpT op, Operation *newOp, ArrayAttr params, Block *body,
    StringRef expectedLaunchContext, SmallVectorImpl<StringAttr> &paramNames) {
  if (!params)
    return success();
  paramNames.reserve(params.size());
  MLIRContext *ctx = op.getContext();
  for (Attribute p : params) {
    auto dict = dyn_cast<DictionaryAttr>(p);
    if (!dict)
      return op.emitOpError("invalid parameters entry");
    auto name = dict.template getAs<StringAttr>("name");
    if (!name)
      return op.emitOpError("parameters entry missing `name`");
    unsigned index = paramNames.size();
    Type paramType = undef;
    if (std::optional<StringRef> launchContext =
            getLaunchContextParameterKind(dict)) {
      if (failed(validateLaunchContextParameter(op.getOperation(), dict, index,
                                                *launchContext,
                                                expectedLaunchContext)))
        return failure();
      paramType = isa<HCWorkitemRegionOp>(newOp)
                      ? Type(WorkitemType::get(ctx, launchGroupShape,
                                               launchSubgroupSize))
                  : isa<HCSubgroupRegionOp>(newOp)
                      ? Type(SubgroupType::get(ctx, launchGroupShape,
                                               launchSubgroupSize))
                      : undef;
    } else if (index == 0) {
      return op.emitOpError(
                 "first nested region parameter must be marked as a ")
             << expectedLaunchContext << " launch context";
    }
    body->addArgument(paramType, op.getLoc());
    paramNames.push_back(name);
  }
  return success();
}

// Walk the source body and lower it into the freshly built `newOp`'s body
// block. For tail-return regions the trailing return is intercepted and
// re-emitted as `hc.yield` to wire up the SSA result handoff; everything
// else goes through the standard op dispatch.
template <typename FrontRegionOpT>
LogicalResult Lowerer::lowerCapturingRegionBody(FrontRegionOpT op,
                                                Operation *newOp,
                                                bool isTailReturnRegion) {
  for (Operation &child : llvm::make_early_inc_range(op.getBody().front())) {
    if (isTailReturnRegion) {
      if (auto retOp = dyn_cast<hc_front::ReturnOp>(&child)) {
        FailureOr<SmallVector<Value>> loweredValues = lowerReturnValues(retOp);
        if (failed(loweredValues))
          return failure();
        if (loweredValues->size() != newOp->getNumResults())
          return retOp.emitOpError(
              "return arity mismatch for tail-return region");
        HCYieldOp::create(builder, retOp.getLoc(), *loweredValues);
        continue;
      }
    }
    if (failed(lowerOp(&child)))
      return failure();
  }
  return success();
}

// Compute the result-type vector for the freshly built region: tail-return
// regions carry one result per source-return value (typed `undef`, refined
// later); plain regions have no results.
template <typename FrontRegionOpT>
FailureOr<SmallVector<Type>>
Lowerer::capturingRegionResultTypes(FrontRegionOpT op,
                                    bool isTailReturnRegion) {
  if (!isTailReturnRegion)
    return SmallVector<Type>{};
  FailureOr<unsigned> arityOr = captureRegionTailReturnArity(op);
  if (failed(arityOr))
    return failure();
  SmallVector<Type> resultTypes;
  resultTypes.assign(*arityOr, undef);
  return resultTypes;
}

template <typename HCRegionOpT, typename FrontRegionOpT>
LogicalResult Lowerer::lowerCapturingRegion(FrontRegionOpT op) {
  // `hc.{workitem,subgroup}_region` match `hc_front` 1:1 (captures +
  // body). Nested-def folding is a later pass; if the front op declares
  // parameters we still add them as block args, plus one leading
  // `hc.assign "<p>", %arg` per param so the body's name lookups
  // resolve via the promotion pass. The `hc` op carries only a
  // captures list (no formal params), matching ODS.
  bool sourceEmpty = op.getBody().empty();
  if (!sourceEmpty && !op.getBody().hasOneBlock())
    return op.emitOpError("hc_front nested region must be single-block");

  // Folded `return inner()` regions lower to ordinary SSA control flow:
  // yield from the nested scope, then return from the enclosing callable.
  bool isTailReturnRegion = op.getTailReturnAttr() != nullptr;
  FailureOr<SmallVector<Type>> resultTypesOr =
      capturingRegionResultTypes(op, isTailReturnRegion);
  if (failed(resultTypesOr))
    return failure();

  auto newOp = HCRegionOpT::create(builder, op.getLoc(), *resultTypesOr,
                                   op.getCapturesAttr());
  Block *body = new Block();
  StringRef expectedLaunchContext =
      isa<HCWorkitemRegionOp>(newOp.getOperation()) ? StringRef("workitem")
                                                    : StringRef("subgroup");
  SmallVector<StringAttr> paramNames;
  if (failed(populateCapturingRegionParams(
          op, newOp.getOperation(),
          op->template getAttrOfType<ArrayAttr>("parameters"), body,
          expectedLaunchContext, paramNames)))
    return failure();
  newOp.getBody().push_back(body);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    for (auto [idx, name] : llvm::enumerate(paramNames))
      HCAssignOp::create(builder, op.getLoc(), name, body->getArgument(idx));
    if (!sourceEmpty && failed(lowerCapturingRegionBody(
                            op, newOp.getOperation(), isTailReturnRegion)))
      return failure();
  }

  if (!isTailReturnRegion)
    return success();
  // This is emitted in the enclosing block after the newly built region op.
  HCReturnOp::create(builder, op.getLoc(), newOp->getResults());
  return success();
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

FailureOr<CallArgs> Lowerer::collectCallArgs(hc_front::CallOp op) {
  CallArgs args;
  for (Value arg : op.getArguments()) {
    auto kwIt = keywordInfo.find(arg);
    if (kwIt != keywordInfo.end()) {
      // Keyword arguments stay as SSA values at this boundary. Intrinsic
      // const-kwargs promote selected `hc.const` values back to attributes
      // later, and load/vload shape kwargs now remain ordinary operands.
      args.kwvalues[kwIt->second.name] = kwIt->second.loweredValue;
      continue;
    }
    FailureOr<Value> loweredOr =
        lowerValueOperand(arg, op.getOperation(), "call argument");
    if (failed(loweredOr))
      return failure();
    Value lowered = *loweredOr;
    if (!lowered) {
      op.emitOpError("call argument did not lower to an hc value");
      return failure();
    }
    args.positional.push_back(lowered);
  }
  return args;
}

// Strip the leading "@" the Python driver stamps so we can hand a bare
// symbol name to `FlatSymbolRefAttr::get` (which re-adds it).
static StringRef stripLeadingAt(StringRef s) {
  return s.starts_with("@") ? s.drop_front() : s;
}

// Compute the result-type vector for an `hc.call`. `hc` calls usually
// carry one progressive-typing result. Helpers annotated `-> None` are
// side-effect only, so their call sites carry no result.
SmallVector<Type> Lowerer::deduceCalleeResultTypes(hc_front::CallOp op,
                                                   FlatSymbolRefAttr symRef,
                                                   StringRef callee) {
  SmallVector<Type> resultTypes;
  if (auto decl = SymbolTable::lookupNearestSymbolFrom<HCFuncOp>(
          op.getOperation(), symRef)) {
    if (std::optional<FunctionType> fnType = decl.getFunctionType())
      resultTypes.assign(fnType->getResults().begin(),
                         fnType->getResults().end());
    else
      resultTypes.push_back(undef);
    return resultTypes;
  }
  if (!unresolvedFrontFuncDeclaresNoResults(op.getOperation(), callee))
    resultTypes.push_back(undef);
  return resultTypes;
}

// `hc_front.call` to a declared `hc.func`: the lowered op is a plain
// `hc.call` against the symbol with the positional operands.
FailureOr<Value> Lowerer::lowerCalleeCall(hc_front::CallOp op,
                                          FlatSymbolRefAttr symRef,
                                          StringRef callee,
                                          const CallArgs &args) {
  SmallVector<Type> resultTypes = deduceCalleeResultTypes(op, symRef, callee);
  auto call = HCCallOp::create(builder, op.getLoc(), resultTypes, symRef,
                               args.positional);
  if (call->getNumResults() == 0)
    return Value();
  return call.getResult(0);
}

namespace {
struct IntrinsicCallSets {
  ArrayAttr declaredParameters;
  ArrayAttr constKwargsAttr;
  ArrayAttr keywordOnlyAttr;
  llvm::SmallDenseSet<StringRef> constKwargSet;
  llvm::SmallDenseSet<StringRef> keywordOnlySet;
};

static IntrinsicCallSets buildIntrinsicCallSets(HCIntrinsicOp intrDecl) {
  IntrinsicCallSets s;
  s.declaredParameters = intrDecl.getParametersAttr();
  s.constKwargsAttr = intrDecl.getConstKwargsAttr();
  s.keywordOnlyAttr = intrDecl.getKeywordOnlyAttr();
  if (s.constKwargsAttr)
    for (Attribute kw : s.constKwargsAttr)
      if (auto str = dyn_cast<StringAttr>(kw))
        s.constKwargSet.insert(str.getValue());
  if (s.keywordOnlyAttr)
    for (Attribute kw : s.keywordOnlyAttr)
      if (auto str = dyn_cast<StringAttr>(kw))
        s.keywordOnlySet.insert(str.getValue());
  return s;
}

// Diagnose unknown kwargs / kwattrs at the call site once we've finished
// matching declared parameters and the const-kwarg whitelist.
static LogicalResult validateRemainingIntrinsicKwargs(
    hc_front::CallOp op, StringRef callee, const CallArgs &args,
    const llvm::SmallDenseSet<StringRef> &consumedKwargs,
    const llvm::SmallDenseSet<StringRef> &constKwargSet) {
  for (auto &kv : args.kwvalues) {
    StringRef name = kv.first();
    if (consumedKwargs.contains(name) || constKwargSet.contains(name))
      continue;
    op.emitOpError("intrinsic '@")
        << callee << "' called with unknown keyword argument '" << name << "'";
    return failure();
  }
  for (auto &kv : args.kwattrs) {
    StringRef name = kv.first();
    if (!constKwargSet.contains(name)) {
      op.emitOpError("intrinsic '@")
          << callee << "' called with unknown keyword attribute '" << name
          << "'";
      return failure();
    }
  }
  return success();
}
} // namespace

namespace {
// Read the declared parameter name at `i`, surfacing a precise diagnostic
// if the entry isn't a `StringAttr`.
static FailureOr<StringRef> declaredParamName(hc_front::CallOp op,
                                              StringRef callee,
                                              ArrayAttr params, unsigned i) {
  auto nameAttr = dyn_cast<StringAttr>(params[i]);
  if (!nameAttr) {
    op.emitOpError("intrinsic '@")
        << callee << "' has malformed `parameters` entry at index " << i
        << ": expected StringAttr, got " << params[i];
    return failure();
  }
  return nameAttr.getValue();
}

// Verify that none of the positional operands binds a keyword-only
// parameter slot.
static LogicalResult validatePositionalSlots(hc_front::CallOp op,
                                             StringRef callee, ArrayAttr params,
                                             const IntrinsicCallSets &sets,
                                             size_t positionalCount) {
  for (unsigned i = 0; i < positionalCount; ++i) {
    FailureOr<StringRef> nameOr = declaredParamName(op, callee, params, i);
    if (failed(nameOr))
      return failure();
    if (sets.keywordOnlySet.contains(*nameOr)) {
      op.emitOpError("intrinsic '@")
          << callee << "' parameter '" << *nameOr
          << "' is keyword-only and cannot be passed positionally";
      return failure();
    }
  }
  return success();
}

// Bind a single non-positional declared parameter against the call's
// kwargs. Skips const_kwargs (handled elsewhere) and pushes the matching
// SSA value into `operands` while recording consumption in
// `consumedKwargs`.
static LogicalResult
bindOneIntrinsicKwarg(hc_front::CallOp op, StringRef callee,
                      const CallArgs &args, const IntrinsicCallSets &sets,
                      StringRef pname, SmallVectorImpl<Value> &operands,
                      llvm::SmallDenseSet<StringRef> &consumedKwargs) {
  if (sets.constKwargSet.contains(pname))
    return success();
  // Intrinsic IR reserves `hc_front.keyword` operands for Python
  // keyword-only parameters. Positional-or-keyword spelling would make
  // hand-authored IR depend on source argument order rather than the
  // callee's declared ABI.
  if (sets.keywordOnlyAttr && !sets.keywordOnlySet.contains(pname)) {
    if (args.kwvalues.contains(pname)) {
      op.emitOpError("intrinsic '@")
          << callee << "' parameter '" << pname
          << "' is positional and cannot be passed as a keyword";
      return failure();
    }
    op.emitOpError("intrinsic '@")
        << callee << "' missing required positional argument '" << pname << "'";
    return failure();
  }
  auto valIt = args.kwvalues.find(pname);
  if (valIt == args.kwvalues.end()) {
    op.emitOpError("intrinsic '@")
        << callee << "' missing required kwarg '" << pname << "'";
    return failure();
  }
  operands.push_back(valIt->second);
  consumedKwargs.insert(pname);
  return success();
}

// Walk the callee's declared parameter list to anchor operand order:
// positionals bind only the prefix before any keyword-only marker,
// keyword operands bind only keyword-only slots, and const_kwargs are
// skipped (they land as call-site attributes). Matching by declared-
// parameter index rather than `kwvalues` iteration keeps the order
// deterministic across hash layouts.
static LogicalResult
bindIntrinsicOperands(hc_front::CallOp op, StringRef callee,
                      const CallArgs &args, const IntrinsicCallSets &sets,
                      SmallVectorImpl<Value> &operands,
                      llvm::SmallDenseSet<StringRef> &consumedKwargs) {
  if (!sets.declaredParameters) {
    if (args.kwvalues.empty())
      return success();
    // Keyword operands need the callee's full ordered parameter list to
    // anchor their SSA slots. Declaration-only hand-written intrinsics
    // without `parameters` can still accept positional calls, but there is
    // no deterministic kwarg order to recover.
    op.emitOpError("intrinsic '@")
        << callee
        << "' call has keyword arguments but callee declares no "
           "`parameters` — the hc_front driver must stamp them";
    return failure();
  }
  ArrayAttr params = sets.declaredParameters;
  if (operands.size() > params.size()) {
    op.emitOpError("intrinsic '@") << callee << "' declares " << params.size()
                                   << " parameter(s), call site supplies "
                                   << operands.size() << " positional";
    return failure();
  }
  if (failed(
          validatePositionalSlots(op, callee, params, sets, operands.size())))
    return failure();
  for (unsigned i = operands.size(), e = params.size(); i < e; ++i) {
    FailureOr<StringRef> pnameOr = declaredParamName(op, callee, params, i);
    if (failed(pnameOr))
      return failure();
    if (failed(bindOneIntrinsicKwarg(op, callee, args, sets, *pnameOr, operands,
                                     consumedKwargs)))
      return failure();
  }
  return success();
}

// Promote declared const kwargs to call-site attributes from the producing
// `hc.const` payload. Const-kwargs must be constant, and hc.const is the
// canonical producer; fish the payload attribute off the const op so the
// kwarg lands as an attribute, not an SSA operand.
static void applyConstKwargAttrs(Operation *call, ArrayAttr constKwargsAttr,
                                 const CallArgs &args) {
  if (!constKwargsAttr)
    return;
  for (Attribute kw : constKwargsAttr) {
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
    if (auto constOp = valIt->second.getDefiningOp<HCConstOp>())
      call->setAttr(kwName, constOp.getValue());
  }
}
} // namespace

FailureOr<Value> Lowerer::lowerIntrinsicCall(hc_front::CallOp op,
                                             FlatSymbolRefAttr symRef,
                                             StringRef callee,
                                             const CallArgs &args) {
  // The lowered `hc.intrinsic` symbol carries both the full declared
  // parameter order and the const-kwarg whitelist. Intrinsic
  // declarations are materialized before caller bodies are lowered, so
  // this lookup is independent of source order.
  HCIntrinsicOp intrDecl = SymbolTable::lookupNearestSymbolFrom<HCIntrinsicOp>(
      op.getOperation(), symRef);
  if (!intrDecl) {
    op.emitOpError("intrinsic '@")
        << callee << "' does not resolve to a lowered hc.intrinsic";
    return failure();
  }
  IntrinsicCallSets sets = buildIntrinsicCallSets(intrDecl);
  SmallVector<Value> operands(args.positional.begin(), args.positional.end());
  llvm::SmallDenseSet<StringRef> consumedKwargs;
  if (failed(bindIntrinsicOperands(op, callee, args, sets, operands,
                                   consumedKwargs)))
    return failure();
  if (failed(validateRemainingIntrinsicKwargs(op, callee, args, consumedKwargs,
                                              sets.constKwargSet)))
    return failure();
  auto call = HCCallIntrinsicOp::create(builder, op.getLoc(), TypeRange{undef},
                                        symRef, operands);
  applyConstKwargAttrs(call.getOperation(), sets.constKwargsAttr, args);
  return call.getResult(0);
}

FailureOr<Value> Lowerer::lowerNamedCall(hc_front::CallOp op, RefInfo ref,
                                         StringRef kind, const CallArgs &args) {
  StringRef callee = ref.getString("callee");
  if (callee.empty()) {
    op.emitOpError("`ref.callee` missing for ") << kind << " name ref";
    return failure();
  }
  callee = stripLeadingAt(callee);
  auto symRef = FlatSymbolRefAttr::get(op.getContext(), callee);
  if (kind == "callee")
    return lowerCalleeCall(op, symRef, callee, args);
  return lowerIntrinsicCall(op, symRef, callee, args);
}

// Diagnose surviving `inline`/`local` calls — both should have been
// consumed by upstream passes (`-hc-front-inline` and
// `-hc-front-fold-region-defs` respectively). Surfacing them here as a
// loud, located error catches pipeline misordering.
static FailureOr<Value> diagnoseUnconsumedFrontCallKind(hc_front::CallOp op,
                                                        StringRef kind) {
  if (kind == "inline") {
    op.emitOpError("`ref.kind = \"inline\"` call survived to conversion; "
                   "run `-hc-front-inline` before `-convert-hc-front-to-hc`");
    return failure();
  }
  if (kind == "local") {
    // `-hc-front-fold-region-defs` owns the ghost
    // `name{local}+call(+return)` trail Python emits for a
    // `@group.workitems def inner(): ...; inner()` immediate-call
    // shape. A surviving call to a local identifier means the folder
    // didn't run — the region op itself is already the lowering, so
    // there is no callable for us to dispatch against.
    op.emitOpError(
        "`ref.kind = \"local\"` call survived to conversion; run "
        "`-hc-front-fold-region-defs` before `-convert-hc-front-to-hc`");
    return failure();
  }
  op.emitOpError("unsupported callee ref.kind '") << kind << "'";
  return failure();
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
  RefInfo ref = RefInfo::get(nameOp);
  // `lowerName` already diagnosed a malformed `ref` on `nameOp` before we
  // got here (SSA: def visited before use). Re-check anyway so this
  // function's preconditions don't quietly rely on traversal order.
  if (failed(ref.diagnoseIfMalformed(nameOp)))
    return failure();
  StringRef kind = ref.getKind();

  // Layout primitives (today: `as_layout(value, descriptor)`) want their
  // descriptor argument inspected as a captured `hc_front.name` rather
  // than lowered to an SSA value — the descriptor maps to a null entry
  // in `valueMap`, and `collectCallArgs` would reject it as "did not
  // lower". Dispatch early.
  if (kind == "layout_op")
    return lowerLayoutOpCall(op, ref);

  FailureOr<CallArgs> argsOr = collectCallArgs(op);
  if (failed(argsOr))
    return failure();
  CallArgs &args = *argsOr;

  if (kind == "callee" || kind == "intrinsic")
    return lowerNamedCall(op, ref, kind, args);
  if (kind == "builtin")
    // Consumed-by-parent builtins (right now: `range`, always folded by the
    // for-loop lowering). No stand-alone hc op. Returning a null Value is
    // safe because the only consumers walk the original hc_front op.
    return Value();
  return diagnoseUnconsumedFrontCallKind(op, kind);
}

// Method-name buckets for DSL-method dispatch. Keep these in one place
// so adding a new builtin is a single-row edit.
static bool isUnaryBaseMethod(StringRef method) {
  return method == "vec" || method == "with_inactive" || method == "astype";
}

static bool isMemOpMethod(StringRef method) {
  static constexpr StringRef kMemOps[] = {"load",  "vload", "store", "vzeros",
                                          "vones", "vfull", "zeros", "ones",
                                          "full",  "empty"};
  return llvm::is_contained(kMemOps, method);
}

// Reductions live on every tensor-shaped value (the simulator surfaces
// them on `_MaskedValue` and `np.<reducer>` on the carrier shape).
// `prod` is listed here so the dispatcher can pin a clear "not yet
// wired" diagnostic instead of leaking through as an unknown method.
static bool isReduceMethod(StringRef method) {
  return method == "sum" || method == "max" || method == "min" ||
         method == "prod";
}

// `np.<func>(...)` calls the front-to-hc rewrite forwards to a
// `hc.builtin_call "numpy.<func>"` carrier. Dispatch is name-based —
// every entry here is an opaque-from-the-dialect's-POV NumPy ufunc that
// downstream lowering must recognise. We don't restrict to unary because
// `np.maximum(a, b)` etc. share the same shape; arity is enforced
// downstream (or by the lowering pattern that consumes the call).
//
// Why a per-bead allow-list and not "any numpy attr": a typo
// (`np.srqt`) would otherwise silently lower to a builtin_call with a
// bogus name that only fails at the lowering pass, far from the
// source. Failing here keeps the diagnostic pointed at the user-visible
// call. New entries cost one row plus a downstream lowering pattern.
static bool isNumpyBuiltinCallMethod(StringRef method) {
  return method == "sqrt" || method == "exp";
}

FailureOr<Value> Lowerer::lowerDslMethodCall(hc_front::CallOp call,
                                             hc_front::AttrOp attr) {
  RefInfo ref = RefInfo::get(attr);
  // Same defense-in-depth as `lowerCall`: `lowerAttr` has already run on
  // `attr` in a well-formed walk, but this re-check keeps the dispatch
  // independent of traversal order.
  if (failed(ref.diagnoseIfMalformed(attr)))
    return failure();

  // `hc_front.attr`'s `$name` is the authoritative method spelling. The
  // resolver stamps `ref = {kind = "dsl_method", method = "<name>"}` only
  // when the base was classifiable; chained attrs (`a[i].vec()`,
  // `buf.vec().with_inactive(...)`) land on an `hc_front.subscript` or
  // `hc_front.call` base that `_classify_attr` leaves unclassified, so no
  // `ref.method` gets stamped. Reading the method name off the op itself
  // makes dispatch work regardless of whether the resolver reached this
  // site; any `ref.method` stamp is redundant and we don't cross-check.
  StringRef method = attr.getName();

  FailureOr<CallArgs> argsOr = collectCallArgs(call);
  if (failed(argsOr))
    return failure();
  CallArgs &args = *argsOr;

  // Module-namespace attr bases — numpy as a module isn't a lowerable
  // value, so these branches dispatch without lowering `attr.getBase()`
  // (which would null-fail). Every other attr base goes through the
  // value-base bucket dispatch.
  if (ref.getKind() == "numpy_dtype_type")
    return lowerNumpyDtypeCall(call, ref, args);
  if (ref.getKind() == "numpy_attr" && isNumpyBuiltinCallMethod(method))
    return lowerNumpyBuiltinCall(call, method, args);
  return lowerValueBaseMethodCall(call, attr, method, args);
}

FailureOr<Value> Lowerer::lowerValueBaseMethodCall(hc_front::CallOp call,
                                                   hc_front::AttrOp attr,
                                                   StringRef method,
                                                   const CallArgs &args) {
  FailureOr<Value> baseOr =
      lowerValueOperand(attr.getBase(), call.getOperation(), "method base");
  if (failed(baseOr))
    return failure();
  Value base = *baseOr;

  if (isUnaryBaseMethod(method))
    return lowerUnaryBaseMethod(call, method, base, args);
  if (isReduceMethod(method)) {
    if (!base) {
      call.emitOpError(method) << ": base did not lower";
      return failure();
    }
    return lowerReduceMethod(call, method, base, args);
  }
  if (isMemOpMethod(method))
    return lowerMemOp(call, method, args);
  if (Value lowered = tryLowerLaunchGeoCall(call, method, args))
    return {lowered};

  call.emitOpError("unsupported dsl_method '") << method << "'";
  return failure();
}

// Layout descriptor sentinel: ``as_layout(value, None)`` is the
// user-marked boundary that drops the value's layout (the operand's
// wave-wide / broadcast addressing convenience stops applying past
// this point — the result is a standalone bare carrier whose
// effective span is the dim product). Recognize the descriptor here
// by chasing back to the ``hc_front.constant`` producer; the
// emitter stamps ``python_kind = "NoneType"`` on the constant when
// the Python literal was ``None``.
static bool isAsLayoutNoneSentinel(Value descriptor) {
  auto constOp = descriptor.getDefiningOp<hc_front::ConstantOp>();
  if (!constOp)
    return false;
  auto kind = constOp->getAttrOfType<StringAttr>("python_kind");
  return kind && kind.getValue() == "NoneType";
}

// Validate the call shape `as_layout(value, descriptor, *, shape=?)`
// — exactly two positionals, optional `shape=` kwarg, nothing else.
// Tensor / vector callers (the verifier rejects `shape=` on those
// flavors) must leave `shape=` absent; pointer-rooted (`!hc.buffer`)
// callers use it to declare the layout's reinterpreted extent.
LogicalResult Lowerer::validateAsLayoutCallShape(hc_front::CallOp op,
                                                 ValueRange args) {
  // Drop kwargs before counting positionals — the same call may carry
  // a `shape=` kwarg, and `args.size()` includes both kinds.
  unsigned positional = 0;
  bool seenShape = false;
  for (Value v : args) {
    if (keywordInfo.contains(v)) {
      auto kw = v.getDefiningOp<hc_front::KeywordOp>();
      if (kw && kw.getName() == "shape") {
        if (seenShape) {
          op.emitOpError("as_layout `shape=` kwarg appears more than once");
          return failure();
        }
        seenShape = true;
        continue;
      }
      op.emitOpError("as_layout accepts only `shape=` as a keyword "
                     "argument; got `")
          << (kw ? kw.getName() : StringRef{"<unknown>"}) << "=`";
      return failure();
    }
    ++positional;
  }
  if (positional != 2) {
    op.emitOpError("as_layout expects 2 positional arguments "
                   "(value, layout descriptor); got ")
        << positional;
    return failure();
  }
  return success();
}

Lowerer::AsLayoutCallSplit Lowerer::splitAsLayoutCallArgs(ValueRange args) {
  AsLayoutCallSplit out;
  for (Value arg : args) {
    auto kwIt = keywordInfo.find(arg);
    if (kwIt == keywordInfo.end()) {
      out.positional.push_back(arg);
      continue;
    }
    if (kwIt->second.name == "shape")
      out.shapeKwValue = kwIt->second.loweredValue;
  }
  return out;
}

FailureOr<Value> Lowerer::lowerLayoutOpCall(hc_front::CallOp op,
                                            const RefInfo &ref) {
  // Today the only ``layout_op`` primitive is ``as_layout(value,
  // descriptor, *, shape=?)``. The dispatch is keyed on the ref's `op`
  // string so we can grow more primitives (`as_dense`, mask-rebinders,
  // ...) without touching this switch's neighbors.
  StringRef opName = ref.getString("op");
  if (opName != "as_layout") {
    op.emitOpError("unsupported layout_op '") << opName << "'";
    return failure();
  }

  ValueRange args = op.getArguments();
  if (failed(validateAsLayoutCallShape(op, args)))
    return failure();

  AsLayoutCallSplit split = splitAsLayoutCallArgs(args);

  FailureOr<Value> valueOr = lowerValueOperand(
      split.positional[0], op.getOperation(), "as_layout value");
  if (failed(valueOr))
    return failure();
  Value value = *valueOr;
  if (!value) {
    op.emitOpError("as_layout value did not lower to an hc value");
    return failure();
  }

  if (isAsLayoutNoneSentinel(split.positional[1])) {
    if (split.shapeKwValue) {
      op.emitOpError("as_layout(value, None) is the strip-layout "
                     "boundary and does not accept `shape=`");
      return failure();
    }
    return {HCStripLayoutOp::create(builder, op.getLoc(), undef, value)
                .getResult()};
  }

  FailureOr<LayoutAttr> layout = readLayoutFromValue(
      split.positional[1], op.getOperation(), "as_layout layout");
  if (failed(layout))
    return failure();

  return {HCAsLayoutOp::create(builder, op.getLoc(), undef, value,
                               split.shapeKwValue, *layout)
              .getResult()};
}

FailureOr<Value> Lowerer::lowerNumpyDtypeCall(hc_front::CallOp call,
                                              const RefInfo &ref,
                                              const CallArgs &args) {
  // `np.<dtype>(lit)` — Python's value-constructor form for numpy
  // scalar types — shows up as a call whose callee is an attr
  // classified as `numpy_dtype_type`. The attr's `ref.dtype` names
  // the destination type; the single positional literal supplies the
  // payload. Emit a fresh `hc.const` carrying a typed `FloatAttr` /
  // `IntegerAttr` so consumers like `hc.with_inactive` receive an ordinary
  // scalar SSA value (not a dtype handle).
  //
  // Degradation path: no positional arg still means "the dtype handle",
  // and an uncoercible literal (NaN/Inf or out-of-range float -> int) still
  // falls back to the `hc.const <TypeAttr>` materialized by `lowerAttr` so
  // downstream users diagnose their own payload expectations. A non-literal
  // positional is different: silently returning the dtype handle hides the
  // bad source shape, so diagnose it here and point users at `.astype`.
  StringRef dtype = ref.getString("dtype");
  Attribute typed;
  if (!args.positional.empty()) {
    if (auto litOp = args.positional.front().getDefiningOp<HCConstOp>()) {
      if (auto tyOpt = resolveNumpyDtypeType(call.getContext(), dtype))
        typed = coerceNumpyLiteral(*tyOpt, litOp.getValue());
    } else {
      call.emitOpError("numpy dtype constructor `np.")
          << dtype
          << "(...)` only accepts literal positional arguments in hc_front; "
             "use `value.astype(np."
          << dtype << ")` for SSA values";
      return failure();
    }
  }
  if (typed)
    return {
        HCConstOp::create(builder, call.getLoc(), undef, typed).getResult()};
  return lowerValueOperand(call.getCallee(), call.getOperation(),
                           "dtype callee");
}

// `np.<func>(...)` forwarded to `hc.builtin_call`. Caller has gated on
// `isNumpyBuiltinCallMethod`. Arity / kwarg policing happens
// downstream where the lowering pattern knows what `numpy.<method>`
// expects; here we just refuse keyword arguments (no ufunc in our
// surface uses them) and confirm every positional lowered.
FailureOr<Value> Lowerer::lowerNumpyBuiltinCall(hc_front::CallOp call,
                                                StringRef method,
                                                const CallArgs &args) {
  if (!args.kwattrs.empty() || !args.kwvalues.empty()) {
    call.emitOpError("`np.") << method << "(...)` takes no keyword arguments";
    return failure();
  }
  SmallVector<Value> operands;
  operands.reserve(args.positional.size());
  for (auto [index, operand] : llvm::enumerate(args.positional)) {
    if (!operand) {
      call.emitOpError("`np.")
          << method << "(...)`: argument " << index << " did not lower";
      return failure();
    }
    operands.push_back(operand);
  }
  // `numpy.<method>` — dotted so the lowering can dispatch on the full
  // path. New library namespaces (`math.*`, ...) compose the same way.
  std::string qualified = ("numpy." + method).str();
  StringAttr name = builder.getStringAttr(qualified);
  return HCBuiltinCallOp::create(builder, call.getLoc(), undef, name, operands)
      .getResult();
}

// Lower `base.vec(layout=?)`. Optional `layout=` kwarg threads through
// to the resulting `hc.vec`.
static FailureOr<Value> lowerVecMethod(OpBuilder &builder, Type undef,
                                       hc_front::CallOp call, Value base) {
  FailureOr<LayoutAttr> layout = consumeLayoutKwarg(call);
  if (failed(layout))
    return failure();
  return {HCVecOp::create(builder, call.getLoc(), undef, base, *layout)
              .getResult()};
}

// Lower `base.with_inactive(value=v)`. The `value=` kwarg is required.
static FailureOr<Value> lowerWithInactiveMethod(OpBuilder &builder, Type undef,
                                                hc_front::CallOp call,
                                                Value base,
                                                const CallArgs &args) {
  auto valIt = args.kwvalues.find("value");
  if (valIt == args.kwvalues.end()) {
    call.emitOpError("with_inactive missing `value=` kwarg");
    return failure();
  }
  Value inactive = valIt->second;
  if (!inactive) {
    call.emitOpError("with_inactive value did not lower to an hc value");
    return failure();
  }
  return {
      HCWithInactiveOp::create(builder, call.getLoc(), undef, base, inactive)
          .getResult()};
}

// Lower `base.astype(target)`. The target must resolve to a TypeAttr
// — either an `hc.const` carrying it directly, or a plain `TypeAttr`
// stashed on the source name op.
static FailureOr<Value> lowerAsTypeMethod(OpBuilder &builder, Type undef,
                                          hc_front::CallOp call, Value base,
                                          const CallArgs &args) {
  if (args.positional.empty()) {
    call.emitOpError("astype missing target type");
    return failure();
  }
  Value target = args.positional.front();
  TypeAttr targetAttr;
  if (auto constOp = target.getDefiningOp<HCConstOp>())
    if (auto t = dyn_cast<TypeAttr>(constOp.getValue()))
      targetAttr = t;
  if (!targetAttr) {
    call.emitOpError("astype target must resolve to a TypeAttr");
    return failure();
  }
  return {HCAsTypeOp::create(builder, call.getLoc(), undef, base, targetAttr)
              .getResult()};
}

// `.sum` / `.max` / `.min` map onto the same `hc.reduce` op with three
// different kinds. Naming the surface methods in one place keeps the
// dispatcher mechanical and `lowerReduceMethod` decoupled from the
// classifier's spelling list.
static std::optional<ReduceKind> reduceMethodKind(StringRef method) {
  if (method == "sum")
    return ReduceKind::Sum;
  if (method == "max")
    return ReduceKind::Max;
  if (method == "min")
    return ReduceKind::Min;
  return std::nullopt;
}

// Walk an SSA value back to the integer literal it represents, or
// fail. Reductions live at the DSL boundary where `axis=N` arrives
// as an `hc.const` over an `IntegerAttr` (the same shape the
// classifier already emits for `np.<dtype>` literals, buffer-dim
// indices, etc.). Anything else — a captured variable, a `range`
// iteration variable, a `.shape[k]` query — leaks an SSA value that
// won't fold here, and the dialect would reject the resulting
// non-attr axis anyway.
static FailureOr<int64_t> tryGetLiteralInt(Value v) {
  auto constOp = v.getDefiningOp<HCConstOp>();
  if (!constOp)
    return failure();
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return failure();
  return intAttr.getInt();
}

// Walk an SSA value back to the bool literal it represents, or
// fail. Mirrors `tryGetLiteralInt` but tolerates the i1-or-i64
// representation `hc.const` allows for boolean literals (the Python
// emitter stamps `True`/`False` as i1 constants today).
static FailureOr<bool> tryGetLiteralBool(Value v) {
  auto constOp = v.getDefiningOp<HCConstOp>();
  if (!constOp)
    return failure();
  if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue()))
    return boolAttr.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getInt() != 0;
  return failure();
}

// Pick the axis SSA value out of `args`, accepting either a single
// positional or an `axis=` kwarg but not both. Returns a null Value
// (still wrapped in success) when no axis is supplied; the caller turns
// that into the surface "axis required" diagnostic.
static FailureOr<Value> pickReduceAxisOperand(hc_front::CallOp call,
                                              StringRef method,
                                              const CallArgs &args) {
  Value axisVal;
  bool fromPositional = false;
  if (!args.positional.empty()) {
    axisVal = args.positional.front();
    fromPositional = true;
    if (args.positional.size() > 1) {
      call.emitOpError(method) << " takes at most one positional argument";
      return failure();
    }
  }
  auto axisKw = args.kwvalues.find("axis");
  if (axisKw != args.kwvalues.end()) {
    if (fromPositional) {
      call.emitOpError(method) << " axis specified twice (positional + axis=)";
      return failure();
    }
    axisVal = axisKw->second;
  }
  return axisVal;
}

// Decode the picked axis SSA value into a non-negative integer. Tuple
// and `None` axis values get bespoke diagnostics — both would also
// fail `tryGetLiteralInt`, but the specific messages tell the user
// which surface form they wrote.
static FailureOr<uint64_t> resolveReduceAxis(hc_front::CallOp call,
                                             StringRef method, Value axisVal) {
  if (axisVal.getDefiningOp<HCTupleOp>()) {
    call.emitOpError(method)
        << " tuple axis is not supported; use a single axis";
    return failure();
  }
  if (auto axisConst = axisVal.getDefiningOp<HCConstOp>())
    if (auto strAttr = dyn_cast<StringAttr>(axisConst.getValue());
        strAttr && strAttr.getValue() == "None") {
      call.emitOpError(method)
          << " axis=None (full-tensor reduction) is not supported";
      return failure();
    }
  FailureOr<int64_t> axisLit = tryGetLiteralInt(axisVal);
  if (failed(axisLit)) {
    call.emitOpError(method) << " axis must be a non-negative integer literal";
    return failure();
  }
  if (*axisLit < 0) {
    call.emitOpError(method) << " axis must be non-negative, got " << *axisLit;
    return failure();
  }
  return static_cast<uint64_t>(*axisLit);
}

// Resolve the optional `keepdims=` kwarg. Absent => `false`; anything
// not a bool literal => diagnose.
static FailureOr<bool> resolveReduceKeepdims(hc_front::CallOp call,
                                             StringRef method,
                                             const CallArgs &args) {
  auto kwIt = args.kwvalues.find("keepdims");
  if (kwIt == args.kwvalues.end())
    return false;
  FailureOr<bool> kdLit = tryGetLiteralBool(kwIt->second);
  if (failed(kdLit)) {
    call.emitOpError(method) << " keepdims must be a bool literal";
    return failure();
  }
  return *kdLit;
}

// Reject unknown kwargs after `axis` / `keepdims` have been consumed.
// A misspelt `axes=` must not silently fall through as a no-axis
// reduction and point the user at the wrong place.
static LogicalResult rejectExtraReduceKwargs(hc_front::CallOp call,
                                             StringRef method,
                                             const CallArgs &args) {
  for (auto &kv : args.kwvalues) {
    StringRef name = kv.first();
    if (name == "axis" || name == "keepdims")
      continue;
    call.emitOpError(method) << " unknown keyword argument '" << name << "'";
    return failure();
  }
  return success();
}

// Lower `base.<sum|max|min>(axis=..., keepdims=?)` to `hc.reduce`. The
// axis can arrive positionally (`.sum(2)`) or by keyword (`.sum(axis=2)`);
// matches the numpy / simulator signature and is the only way to write a
// reduction at the surface today. Keepdims defaults to false. Tuple axis
// and `axis=None` are diagnosed via `resolveReduceAxis` rather than
// silently folded — a full-tensor reduction needs a different lowering,
// and the simulator doesn't accept a tuple form either. `prod` parses as
// a reduction method so the dispatcher routes it here, but the dialect
// doesn't carry the kind yet; reject it explicitly instead of crashing
// on a missing enum case.
FailureOr<Value> Lowerer::lowerReduceMethod(hc_front::CallOp call,
                                            StringRef method, Value base,
                                            const CallArgs &args) {
  std::optional<ReduceKind> kindOpt = reduceMethodKind(method);
  if (!kindOpt) {
    call.emitOpError("reduction method '")
        << method
        << "' is not supported yet; only sum/max/min are wired through "
           "`hc.reduce` today";
    return failure();
  }
  FailureOr<Value> axisOr = pickReduceAxisOperand(call, method, args);
  if (failed(axisOr))
    return failure();
  if (!*axisOr) {
    call.emitOpError(method)
        << " requires an `axis=` argument; full-tensor reductions are not "
           "supported yet";
    return failure();
  }
  FailureOr<uint64_t> axisLit = resolveReduceAxis(call, method, *axisOr);
  if (failed(axisLit))
    return failure();
  FailureOr<bool> keepdims = resolveReduceKeepdims(call, method, args);
  if (failed(keepdims))
    return failure();
  if (failed(rejectExtraReduceKwargs(call, method, args)))
    return failure();
  return {HCReduceOp::create(builder, call.getLoc(), undef, base, *kindOpt,
                             *axisLit, *keepdims)
              .getResult()};
}

FailureOr<Value> Lowerer::lowerUnaryBaseMethod(hc_front::CallOp call,
                                               StringRef method, Value base,
                                               const CallArgs &args) {
  // `x.vec()`, `x.with_inactive(value=...)`, `x.astype(target)` are the
  // mechanical unary-base cases. Anything requiring group/context plumbing
  // (`wi.local_id()`, `group.load(...)`) is handled downstream via the
  // launch-geo fast path.
  // All unary-base DSL methods share the "base did not lower" guard; a
  // classification gap must not let us ship an hc op built on a null
  // operand — the later verifier error would be harder to attribute.
  if (!base) {
    call.emitOpError(method) << ": base did not lower";
    return failure();
  }

  if (method == "vec")
    return lowerVecMethod(builder, undef, call, base);
  if (method == "with_inactive")
    return lowerWithInactiveMethod(builder, undef, call, base, args);
  if (method == "astype")
    return lowerAsTypeMethod(builder, undef, call, base, args);
  llvm_unreachable("unknown unary-base DSL method");
}

// Chained-subscript guard for `load`/`vload`/`store`: if after one peel the
// handle is *still* a `hc.buffer_view`, the user wrote `a[i][j]`-style
// nested subscripts. That lowers to two distinct buffer_views whose index
// lists can't be safely spliced (the outer slice re-indexes the already-
// reduced view, not the original buffer's next axis). Diagnose with the
// rewrite suggestion rather than emit wrong IR or let the rank verifier
// complain about an index count the user didn't write.
static LogicalResult rejectNestedBufferView(hc_front::CallOp call,
                                            StringRef method, Value handle) {
  if (!handle.getDefiningOp<HCBufferViewOp>())
    return success();
  call.emitOpError("chained subscript into `")
      << method
      << "` is not supported; use a single tuple subscript instead "
         "(e.g. `group."
      << method << "(a[i, j], ...)` rather than `a[i][j]`)";
  return failure();
}

// `shape=` is mandatory for the array-init mem ops; surface a precise
// diagnostic when missing.
static FailureOr<Value> requiredShapeKwarg(hc_front::CallOp call,
                                           const CallArgs &args,
                                           StringRef callee) {
  auto shapeIt = args.kwvalues.find("shape");
  if (shapeIt == args.kwvalues.end() || !shapeIt->second) {
    call.emitOpError("`") << callee << "` missing or invalid `shape=` kwarg";
    return failure();
  }
  return shapeIt->second;
}

// `dtype=` is optional. Absent => null `TypeAttr`; present must lower to an
// `HCConstOp` carrying a `TypeAttr`.
static FailureOr<TypeAttr> optionalDtypeKwarg(hc_front::CallOp call,
                                              const CallArgs &args,
                                              StringRef callee) {
  auto dtypeIt = args.kwvalues.find("dtype");
  if (dtypeIt == args.kwvalues.end())
    return TypeAttr();
  Value dtype = dtypeIt->second;
  if (!dtype) {
    call.emitOpError("`") << callee << "` dtype did not lower to an hc value";
    return failure();
  }
  if (auto constOp = dtype.getDefiningOp<HCConstOp>())
    if (auto type = dyn_cast<TypeAttr>(constOp.getValue()))
      return type;
  call.emitOpError("`") << callee << "` dtype must resolve to a TypeAttr";
  return failure();
}

// `load` / `vload`: the frontend passes a (possibly pre-sliced) handle as
// the first positional, remaining positionals go to the indices list, and
// an optional `shape=` kwarg is passed through as a normal SSA operand.
// `peelBufferView` folds the pre-subscripted `hc.buffer_view` into the op's
// own index list; see its banner for the single-level rationale.
FailureOr<Value> Lowerer::lowerMemLoad(hc_front::CallOp call, StringRef method,
                                       const CallArgs &args) {
  if (args.positional.empty()) {
    call.emitOpError("`") << method << "` expects a buffer/tensor argument";
    return failure();
  }
  Value src = args.positional.front();
  SmallVector<Value> indices(args.positional.begin() + 1,
                             args.positional.end());
  src = peelBufferView(src, indices);
  if (failed(rejectNestedBufferView(call, method, src)))
    return failure();
  auto shapeIt = args.kwvalues.find("shape");
  Value shape =
      shapeIt == args.kwvalues.end()
          ? HCUndefValueOp::create(builder, call.getLoc(), undef).getResult()
          : shapeIt->second;
  if (!shape) {
    call.emitOpError("`") << method << "` shape did not lower to an hc value";
    return failure();
  }
  FailureOr<LayoutAttr> layout = consumeLayoutKwarg(call);
  if (failed(layout))
    return failure();
  Operation *op = method == "load"
                      ? HCLoadOp::create(builder, call.getLoc(), undef, src,
                                         indices, shape, *layout)
                            .getOperation()
                      : HCVLoadOp::create(builder, call.getLoc(), undef, src,
                                          indices, shape, *layout)
                            .getOperation();
  return op->getResult(0);
}

// `store` is the one mem op without a shaped result — reinterpreting its
// (absent) output via `layout=` is meaningless, so diagnose at the call
// site instead of silently dropping the kwarg.
FailureOr<Value> Lowerer::lowerMemStore(hc_front::CallOp call,
                                        const CallArgs &args) {
  if (args.positional.size() < 2) {
    call.emitOpError("`store` expects at least (dest, source)");
    return failure();
  }
  if (findKeywordArg(call, "layout")) {
    call.emitOpError("`store` does not accept a `layout=` kwarg "
                     "(stores have no shaped result to relabel)");
    return failure();
  }
  Value dest = args.positional.front();
  Value source = args.positional.back();
  SmallVector<Value> indices(args.positional.begin() + 1,
                             args.positional.end() - 1);
  dest = peelBufferView(dest, indices);
  if (failed(rejectNestedBufferView(call, "store", dest)))
    return failure();
  HCStoreOp::create(builder, call.getLoc(), dest, indices, source, Value{});
  // `hc.store` is a no-result op; success+null signals "consumed, no
  // SSA output" to `lowerCall`, distinct from the failure path below.
  return Value();
}

// `vzeros` / `vones` / `zeros` / `ones` / `empty`: shape + optional dtype +
// optional layout, no fill value.
FailureOr<Value> Lowerer::lowerMemInit(hc_front::CallOp call, StringRef method,
                                       const CallArgs &args) {
  FailureOr<Value> shape = requiredShapeKwarg(call, args, method);
  if (failed(shape))
    return failure();
  FailureOr<TypeAttr> dtype = optionalDtypeKwarg(call, args, method);
  if (failed(dtype))
    return failure();
  FailureOr<LayoutAttr> layout = consumeLayoutKwarg(call);
  if (failed(layout))
    return failure();
  Operation *op = nullptr;
  if (method == "vzeros")
    op = HCVZerosOp::create(builder, call.getLoc(), undef, *shape, *dtype,
                            *layout);
  else if (method == "vones")
    op = HCVOnesOp::create(builder, call.getLoc(), undef, *shape, *dtype,
                           *layout);
  else if (method == "zeros")
    op = HCZerosOp::create(builder, call.getLoc(), undef, *shape, *dtype,
                           *layout);
  else if (method == "ones")
    op = HCOnesOp::create(builder, call.getLoc(), undef, *shape, *dtype,
                          *layout);
  else
    op = HCEmptyOp::create(builder, call.getLoc(), undef, *shape, *dtype,
                           *layout);
  return op->getResult(0);
}

// `vfull` / `full`: same shape/dtype/layout combo as init ops plus a
// `fill_value` operand (sourced from `fill_value=` kwarg or first
// positional).
FailureOr<Value> Lowerer::lowerMemFull(hc_front::CallOp call, StringRef method,
                                       const CallArgs &args) {
  FailureOr<Value> shape = requiredShapeKwarg(call, args, method);
  if (failed(shape))
    return failure();
  FailureOr<TypeAttr> dtype = optionalDtypeKwarg(call, args, method);
  if (failed(dtype))
    return failure();
  Value fill;
  if (auto fillIt = args.kwvalues.find("fill_value");
      fillIt != args.kwvalues.end())
    fill = fillIt->second;
  else if (!args.positional.empty())
    fill = args.positional.front();
  if (!fill) {
    call.emitOpError("`") << method << "` missing `fill_value=` operand";
    return failure();
  }
  FailureOr<LayoutAttr> layout = consumeLayoutKwarg(call);
  if (failed(layout))
    return failure();
  Operation *op = method == "vfull"
                      ? HCVFullOp::create(builder, call.getLoc(), undef, fill,
                                          *shape, *dtype, *layout)
                            .getOperation()
                      : HCFullOp::create(builder, call.getLoc(), undef, fill,
                                         *shape, *dtype, *layout)
                            .getOperation();
  return op->getResult(0);
}

// Bucket-membership predicates for the four mem-op families: keep adjacent
// to `lowerMemOp` so adding a new builtin is a single-row edit.
static bool isMemLoadMethod(StringRef m) { return m == "load" || m == "vload"; }
static bool isMemInitMethod(StringRef m) {
  return m == "vzeros" || m == "vones" || m == "zeros" || m == "ones" ||
         m == "empty";
}
static bool isMemFullMethod(StringRef m) { return m == "vfull" || m == "full"; }

FailureOr<Value> Lowerer::lowerMemOp(hc_front::CallOp call, StringRef method,
                                     const CallArgs &args) {
  if (isMemLoadMethod(method))
    return lowerMemLoad(call, method, args);
  if (method == "store")
    return lowerMemStore(call, args);
  if (isMemInitMethod(method))
    return lowerMemInit(call, method, args);
  if (isMemFullMethod(method))
    return lowerMemFull(call, method, args);
  llvm_unreachable("unknown memory DSL method");
}

Value Lowerer::tryLowerLaunchGeoCall(hc_front::CallOp call, StringRef method,
                                     const CallArgs &args) {
  // `group.{launch_geo}()` with no args is the call form (`wi.local_id()`).
  // The launch-geo op itself is emitted by `lowerAttr` on the call's
  // attr callee; the call is semantically transparent for these
  // getters, so we forward the cached tuple.
  if (!call.getArguments().empty() || !args.kwvalues.empty() ||
      !args.kwattrs.empty())
    return {};
  if (!classifyLaunchGeoMethod(method))
    return {};
  auto attr =
      dyn_cast_if_present<hc_front::AttrOp>(call.getCallee().getDefiningOp());
  if (!attr)
    return {};
  FailureOr<Value> attrValOr = lowerValueOperand(
      attr.getResult(), call.getOperation(), "launch-geo attr");
  return failed(attrValOr) ? Value{} : *attrValOr;
}

static std::optional<unsigned> staticLaunchGeoRequiredRank(Value axisValue) {
  auto constant =
      dyn_cast_if_present<hc_front::ConstantOp>(axisValue.getDefiningOp());
  auto axis =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  if (!axis)
    return std::nullopt;
  int64_t value = axis.getInt();
  if (value < 0 || value >= kMaxLaunchAxis)
    return std::nullopt;
  return static_cast<unsigned>(value + 1);
}

namespace {

// `getitem` against an `attr` directly: tighten the rank for that
// (source, method) pair when the index is a structural constant.
static void noteAttrSubscriptLaunchGeoRank(
    hc_front::SubscriptOp op, hc_front::AttrOp attr,
    llvm::function_ref<void(Value, StringRef, unsigned)> record) {
  StringRef method = attr.getName();
  std::optional<LaunchGeoMethodInfo> methodInfo =
      classifyLaunchGeoMethod(method);
  // `group.shape[N]` lands on the Python alias whose canonical name is
  // `group_shape`. The base hasn't been lowered yet at this pre-walk,
  // so resolve the alias optimistically: the recorded hint is only
  // queried under launch-geo dispatch in `lowerAttr`, and a stray
  // `buffer.shape` entry is dead weight there (the buffer-dim path
  // never consults the rank table).
  if (!methodInfo && method == "shape")
    methodInfo = getLaunchGeoMethodInfo(LaunchGeoMethod::GroupShape);
  if (!methodInfo || methodInfo->isScalar())
    return;
  std::optional<unsigned> rank =
      staticLaunchGeoRequiredRank(op.getIndices().front());
  if (!rank)
    return;
  record(attr.getBase(), method, *rank);
}

// Walks every use of `result` and folds the structural rank of each
// subscript index. Returns `nullopt` when any use isn't a single-index
// constant subscript — i.e. the tuple escapes — so the caller leaves the
// conservative cap-sized fallback in place.
static std::optional<unsigned> allSubscriptUsesRank(Value result) {
  unsigned rank = 0;
  for (Operation *user : result.getUsers()) {
    auto subscript = dyn_cast<hc_front::SubscriptOp>(user);
    if (!subscript || subscript.getBase() != result ||
        subscript.getIndices().size() != 1)
      return std::nullopt;
    std::optional<unsigned> required =
        staticLaunchGeoRequiredRank(subscript.getIndices().front());
    if (!required)
      return std::nullopt;
    rank = std::max(rank, *required);
  }
  return rank;
}

// `getitem` against a call result: only tighten when *every* use of the
// call result is a structural-constant subscript. Any escape (passing the
// tuple by value, dynamic index, etc.) keeps the conservative cap-sized
// fallback intact. The rank is recorded against `attr.getBase()` (the
// launch context SSA value) — the same key used by the property-style
// `noteAttrSubscriptLaunchGeoRank` — so the launch-geo op `lowerAttr`
// emits sees the merged rank regardless of whether the user wrote the
// property or call form.
static void noteCallSubscriptLaunchGeoRank(
    hc_front::CallOp call,
    llvm::function_ref<void(Value, StringRef, unsigned)> record) {
  if (!call.getArguments().empty())
    return;
  auto attr =
      dyn_cast_if_present<hc_front::AttrOp>(call.getCallee().getDefiningOp());
  if (!attr)
    return;
  StringRef method = attr.getName();
  std::optional<LaunchGeoMethodInfo> methodInfo =
      classifyLaunchGeoMethod(method);
  // See `noteAttrSubscriptLaunchGeoRank` for the alias rationale.
  if (!methodInfo && method == "shape")
    methodInfo = getLaunchGeoMethodInfo(LaunchGeoMethod::GroupShape);
  if (!methodInfo || methodInfo->isScalar())
    return;
  std::optional<unsigned> rank = allSubscriptUsesRank(call.getResult());
  if (!rank || *rank == 0)
    return;
  record(attr.getBase(), method, *rank);
}

// Trace an `hc_front.name` (ref = "local") backwards through the most
// recent prior `hc_front.assign` in the same block. Returns the defining
// op of the rhs Value when found, or null otherwise. Reaching-definitions
// across branches / loops is intentionally not modelled: this covers the
// common "bind once, read many times" idiom that lets a launch-geo or
// shape subscript fold through the local binding without us having to
// touch the type of `hc.name_load`. Shared by the static-rank pre-walk
// (`collectStaticLaunchGeometryRanks`) and the subscript fold dispatch
// in `trySubscriptFolds` so both views of a local-bound launch-geo
// tuple stay in sync.
static Operation *traceLocalNameSourceOp(hc_front::NameOp name) {
  RefInfo ref = RefInfo::get(name);
  if (ref.getKind() != "local")
    return nullptr;
  StringRef target = name.getName();
  Block *block = name.getOperation()->getBlock();
  for (Operation *cur = name.getOperation()->getPrevNode(); cur;
       cur = cur->getPrevNode()) {
    if (cur->getBlock() != block)
      break;
    auto assign = dyn_cast<hc_front::AssignOp>(cur);
    if (!assign)
      continue;
    auto tn = dyn_cast_if_present<hc_front::TargetNameOp>(
        assign.getTarget().getDefiningOp());
    if (!tn || tn.getName() != target)
      continue;
    return assign.getValue().getDefiningOp();
  }
  return nullptr;
}

} // namespace

void Lowerer::collectStaticLaunchGeometryRanks(Operation *frontOp) {
  auto record = [&](Value source, StringRef method, unsigned rank) {
    unsigned &current = staticLaunchGeoRanks[source][method];
    current = std::max(current, rank);
  };

  frontOp->walk([&](hc_front::SubscriptOp op) {
    if (op.getIndices().size() != 1)
      return;
    Operation *baseOp = op.getBase().getDefiningOp();
    // Mirror `trySubscriptFolds`: trace local-name bindings to their
    // source so a `gid = group.group_id; gid[0]; gid[1]` idiom still
    // contributes a structural rank hint even though the subscripts
    // sit on the local read rather than on the attr directly.
    bool tracedFromLocal = false;
    if (auto nameOp = dyn_cast_if_present<hc_front::NameOp>(baseOp))
      if (Operation *src = traceLocalNameSourceOp(nameOp)) {
        baseOp = src;
        tracedFromLocal = true;
      }
    if (auto attr = dyn_cast_if_present<hc_front::AttrOp>(baseOp))
      return noteAttrSubscriptLaunchGeoRank(op, attr, record);
    if (auto call = dyn_cast_if_present<hc_front::CallOp>(baseOp)) {
      // Direct `group.method()[N]` keeps the all-uses tightening
      // (a tuple-passed-by-value escape must keep the conservative
      // cap). The local-traced form already used its assignment as a
      // user, so `allSubscriptUsesRank` would refuse to tighten; treat
      // it like the property form instead and record the per-subscript
      // hint against the underlying attr's base.
      if (!tracedFromLocal) {
        noteCallSubscriptLaunchGeoRank(call, record);
        return;
      }
      if (auto attr = dyn_cast_if_present<hc_front::AttrOp>(
              call.getCallee().getDefiningOp()))
        noteAttrSubscriptLaunchGeoRank(op, attr, record);
    }
  });
}

std::optional<unsigned>
Lowerer::getStaticLaunchGeometryRank(Value source, StringRef method) const {
  auto sourceIt = staticLaunchGeoRanks.find(source);
  if (sourceIt == staticLaunchGeoRanks.end())
    return std::nullopt;
  auto methodIt = sourceIt->second.find(method);
  if (methodIt == sourceIt->second.end())
    return std::nullopt;
  return methodIt->second;
}

static LogicalResult checkLaunchGeoSubscript(hc_front::SubscriptOp op,
                                             const LaunchGeoMethodInfo &method,
                                             IntegerAttr axis) {
  if (method.isScalar())
    return op.emitOpError("scalar launch-geo query '")
           << method.name << "' is not subscriptable";

  // Launch-geo axes parameterize the variadic result list built by
  // `tryEmitLaunchGeo`; cap literal axes so a hand-written `group_id[2^31]`
  // cannot allocate a pathological tuple. Dynamic axes are left to
  // `hc.getitem` inference/refinement.
  if (!axis)
    return success();

  int64_t axisValue = axis.getInt();
  if (axisValue < 0 || axisValue >= kMaxLaunchAxis)
    return op.emitOpError("launch-geo axis ")
           << axisValue << " out of range [0, " << kMaxLaunchAxis << ")";
  return success();
}

static unsigned shapeRankOrZero(ShapeAttr shape) {
  return shape ? static_cast<unsigned>(shape.getDims().size()) : 0;
}

unsigned
Lowerer::getLaunchGeometryRank(const LaunchGeoMethodInfo &method,
                               Type contextType,
                               std::optional<unsigned> requiredRank) const {
  auto fallbackRank = [&] {
    return requiredRank.value_or(static_cast<unsigned>(kMaxLaunchAxis));
  };
  ShapeAttr contextWorkShape;
  ShapeAttr contextGroupShape;
  if (std::optional<LaunchContextMetadata> metadata =
          getLaunchContextMetadata(contextType)) {
    contextWorkShape = metadata->workShape;
    contextGroupShape = metadata->groupShape;
  }

  switch (method.rankDomain) {
  case LaunchGeoRankDomain::WorkGridWithGroupFallback:
    return shapeRankOrZero(contextWorkShape)
               ? shapeRankOrZero(contextWorkShape)
               : workRank.value_or(groupRank.value_or(fallbackRank()));
  case LaunchGeoRankDomain::WorkGrid:
    return shapeRankOrZero(contextWorkShape)
               ? shapeRankOrZero(contextWorkShape)
               : workRank.value_or(fallbackRank());
  case LaunchGeoRankDomain::Workgroup:
    // `group_shape` and `work_shape` share a rank by spec --- each
    // workgroup dim aligns one-for-one with a work-grid dim
    // (`doc/langref.md` "logical launch domain"). When the user only
    // declares `work_shape=`, the runtime picks a target-specific
    // default `group_shape` of the same rank; route the per-workgroup
    // queries (`group.shape`, `wi.local_id`, `wi.subgroup_id`) through
    // `workRank` instead of falling all the way to the 32-axis cap so
    // the synthesized op carries the right result count.
    return shapeRankOrZero(contextGroupShape)
               ? shapeRankOrZero(contextGroupShape)
               : groupRank.value_or(workRank.value_or(fallbackRank()));
  case LaunchGeoRankDomain::Scalar:
    return fallbackRank();
  }
  llvm_unreachable("unhandled launch-geometry rank domain");
}

// Emit a per-axis launch-geometry op (group id, local id, etc.) and
// pack its multi-result expansion into an `hc.tuple` so the caller
// just has a single SSA carrier to subscript / store.
template <typename OpT>
static Value emitLaunchGeoMultiResult(OpBuilder &builder, StringRef prefix,
                                      Value context, Location loc,
                                      unsigned rank) {
  FailureOr<SmallVector<Type>> resTypes =
      launchGeometryIdxTypes(builder.getContext(), loc, prefix, rank);
  if (failed(resTypes))
    return {};
  auto op = OpT::create(builder, loc, *resTypes, context);
  auto tupleType = TupleType::get(builder.getContext(), op.getResultTypes());
  return HCTupleOp::create(builder, loc, tupleType, op.getResults());
}

// Single-result launch-geometry op (group_size, wave_size).
template <typename OpT>
static Value emitLaunchGeoScalarResult(OpBuilder &builder, StringRef prefix,
                                       Value context, Location loc) {
  FailureOr<Type> resultType =
      launchGeometryIdxType(builder.getContext(), loc, prefix, 0);
  if (failed(resultType))
    return {};
  auto op = OpT::create(builder, loc, *resultType, context);
  return op.getResult(0);
}

Value Lowerer::tryEmitLaunchGeo(const LaunchGeoMethodInfo &method,
                                Value context, Location loc,
                                std::optional<unsigned> requiredRank) {
  StringRef prefix = method.symbolPrefix;
  unsigned rank =
      getLaunchGeometryRank(method, context.getType(), requiredRank);
  switch (method.method) {
  case LaunchGeoMethod::GroupId:
    return emitLaunchGeoMultiResult<HCGroupIdOp>(builder, prefix, context, loc,
                                                 rank);
  case LaunchGeoMethod::LocalId:
    return emitLaunchGeoMultiResult<HCLocalIdOp>(builder, prefix, context, loc,
                                                 rank);
  case LaunchGeoMethod::SubgroupId:
    return emitLaunchGeoMultiResult<HCSubgroupIdOp>(builder, prefix, context,
                                                    loc, rank);
  case LaunchGeoMethod::GroupShape:
    return emitLaunchGeoMultiResult<HCGroupShapeOp>(builder, prefix, context,
                                                    loc, rank);
  case LaunchGeoMethod::WorkOffset:
    return emitLaunchGeoMultiResult<HCWorkOffsetOp>(builder, prefix, context,
                                                    loc, rank);
  case LaunchGeoMethod::WorkShape:
    return emitLaunchGeoMultiResult<HCWorkShapeOp>(builder, prefix, context,
                                                   loc, rank);
  case LaunchGeoMethod::GroupSize:
    return emitLaunchGeoScalarResult<HCGroupSizeOp>(builder, prefix, context,
                                                    loc);
  case LaunchGeoMethod::WaveSize:
    return emitLaunchGeoScalarResult<HCWaveSizeOp>(builder, prefix, context,
                                                   loc);
  }
  llvm_unreachable("unhandled launch-geometry method");
}

// Walk an `HCConstOp` index back to its `IntegerAttr` payload, which is
// what the launch-geo / buffer-dim folds need to validate the axis.
static IntegerAttr indexConstantAxisAttr(Value idxVal) {
  auto constOp = idxVal ? idxVal.getDefiningOp<HCConstOp>() : nullptr;
  return constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr{};
}

// `base.shape[constant]` -> `hc.buffer_dim`. No cap applies (this is a
// buffer rank, not launch geometry). Returns nullptr unless the static
// shape pattern actually matched.
Value Lowerer::tryLowerShapeSubscript(hc_front::SubscriptOp op, Value baseVal,
                                      IntegerAttr ax) {
  if (!baseVal || !ax)
    return nullptr;
  return HCBufferDimOp::create(
      builder, op.getLoc(), undef, baseVal,
      IntegerAttr::get(IntegerType::get(op.getContext(), 64), ax.getInt()));
}

// Property-style launch-geo subscript fold:
// `base.local_id[N]` -> `hc.getitem(launch-geo-tuple, N)`.
//
// The launch-geo op itself is emitted by `lowerAttr` (a post-order walk
// guarantees the attr has been lowered before this subscript runs); we
// only need to peel the cached tuple here. Tri-state return:
// `failure()` means a diagnostic has already fired (the caller must
// propagate, not fall through to the generic path); `success(null
// Value)` means "didn't match, try the next pattern"; `success(non-null
// Value)` is the lowered result.
FailureOr<Value>
Lowerer::tryLowerLaunchGeoAttrSubscript(hc_front::SubscriptOp op,
                                        hc_front::AttrOp attr, Value idxVal,
                                        IntegerAttr ax) {
  if (!idxVal)
    return Value();
  std::optional<LaunchGeoMethodInfo> methodInfo =
      classifyLaunchGeoMethod(attr.getName());
  // `group.shape[N]` etc. — same alias dispatch as `lowerAttr` uses
  // for the bare attr; see `isLaunchContextFrontParam` for why we
  // walk the front-pass `parameters` dict rather than inspecting the
  // lowered base type.
  if (!methodInfo && attr.getName() == "shape" &&
      isLaunchContextFrontBase(attr.getBase()))
    methodInfo = getLaunchGeoMethodInfo(LaunchGeoMethod::GroupShape);
  if (!methodInfo)
    return Value();
  if (failed(checkLaunchGeoSubscript(op, *methodInfo, ax)))
    return failure();
  FailureOr<Value> attrValOr =
      lowerValueOperand(attr.getResult(), op.getOperation(), "launch-geo attr");
  if (failed(attrValOr))
    return failure();
  Value attrVal = *attrValOr;
  if (!attrVal)
    return Value();
  return Value(
      HCGetItemOp::create(builder, op.getLoc(), undef, attrVal, idxVal));
}

// Property-style `base.method[N]` folds. Tri-state: see
// `tryLowerLaunchGeoAttrSubscript`.
FailureOr<Value> Lowerer::tryLowerAttrSubscript(hc_front::SubscriptOp op,
                                                hc_front::AttrOp attr) {
  if (op.getIndices().size() != 1)
    return Value();
  FailureOr<Value> idxValOr = lowerValueOperand(
      op.getIndices().front(), op.getOperation(), "subscript index");
  if (failed(idxValOr))
    return failure();
  Value idxVal = *idxValOr;
  IntegerAttr ax = indexConstantAxisAttr(idxVal);
  if (attr.getName() == "shape") {
    // `x.shape[N]` lowers two different ways depending on whether the
    // base is a buffer / tensor / vector (→ `hc.buffer_dim`) or a
    // launch-context handle (→ `hc.getitem` against the cached
    // `hc.group_shape` tuple). Disambiguate using the front-pass
    // parameter binding; the lowered base's MLIR type is still the
    // erased `!hc.undef` placeholder at this point.
    if (isLaunchContextFrontBase(attr.getBase()))
      return tryLowerLaunchGeoAttrSubscript(op, attr, idxVal, ax);
    FailureOr<Value> baseValOr =
        lowerValueOperand(attr.getBase(), op.getOperation(), "subscript base");
    if (failed(baseValOr))
      return failure();
    return Value(tryLowerShapeSubscript(op, *baseValOr, ax));
  }
  return tryLowerLaunchGeoAttrSubscript(op, attr, idxVal, ax);
}

// Call-style `base.method()[N]` fold for launch-geo getters: lowers the
// call's result and indexes into it via `hc.getitem`. Tri-state: see
// `tryLowerLaunchGeoAttrSubscript`.
FailureOr<Value> Lowerer::tryLowerCallSubscript(hc_front::SubscriptOp op,
                                                hc_front::CallOp call) {
  auto attr =
      dyn_cast_if_present<hc_front::AttrOp>(call.getCallee().getDefiningOp());
  if (!attr)
    return Value();
  std::optional<LaunchGeoMethodInfo> methodInfo =
      classifyLaunchGeoMethod(attr.getName());
  if (!methodInfo || !call.getArguments().empty() ||
      op.getIndices().size() != 1)
    return Value();
  FailureOr<Value> idxValOr = lowerValueOperand(
      op.getIndices().front(), op.getOperation(), "subscript index");
  if (failed(idxValOr))
    return failure();
  FailureOr<Value> launchGeoOr = lowerValueOperand(
      call.getResult(), op.getOperation(), "launch-geo call result");
  if (failed(launchGeoOr))
    return failure();
  Value idxVal = *idxValOr;
  Value launchGeo = *launchGeoOr;
  if (!launchGeo || !idxVal)
    return Value();
  if (failed(checkLaunchGeoSubscript(op, *methodInfo,
                                     indexConstantAxisAttr(idxVal))))
    return failure();
  return Value(
      HCGetItemOp::create(builder, op.getLoc(), undef, launchGeo, idxVal));
}

// Run the property/call subscript folds, returning tri-state-flattened
// against `lowerSubscript`'s own `Value`-or-null contract: the first
// element says "stop and use this Value (possibly null on diagnosed
// failure)", the second is the lowered Value when present.
Lowerer::SubscriptFoldResult
Lowerer::trySubscriptFolds(hc_front::SubscriptOp op) {
  Operation *baseOp = op.getBase().getDefiningOp();
  // Locally-bound aliases (e.g. `gid = group.work_offset; gid[0]`)
  // route through the same fold path as the inline `group.work_offset[0]`
  // form. The launch-geo / buffer-dim helpers only need the AttrOp or
  // CallOp at the source of the binding, not the `hc.name_load` SSA
  // value that the local read lowers to.
  if (auto nameOp = dyn_cast_if_present<hc_front::NameOp>(baseOp))
    if (Operation *src = traceLocalNameSourceOp(nameOp))
      baseOp = src;
  if (auto attr = dyn_cast_if_present<hc_front::AttrOp>(baseOp)) {
    FailureOr<Value> lowered = tryLowerAttrSubscript(op, attr);
    if (failed(lowered))
      return {true, nullptr};
    if (*lowered)
      return {true, *lowered};
  }
  if (auto call = dyn_cast_if_present<hc_front::CallOp>(baseOp)) {
    FailureOr<Value> lowered = tryLowerCallSubscript(op, call);
    if (failed(lowered))
      return {true, nullptr};
    if (*lowered)
      return {true, *lowered};
  }
  return {false, nullptr};
}

// NumPy `None` / `np.newaxis` subscript sentinel. The frontend emits
// `hc_front.constant<"None"> {python_kind = "NoneType"}` for every
// `x[..., None, ...]` slot; recognize it here so the buffer_view never
// receives the resulting `!hc.undef` value as an index. Same shape as
// `isAsLayoutNoneSentinel`, but kept separate because the surrounding
// validation differs.
static bool isFrontNoneSubscript(Value frontIdx) {
  auto constOp = frontIdx.getDefiningOp<hc_front::ConstantOp>();
  if (!constOp)
    return false;
  auto kind = constOp->getAttrOfType<StringAttr>("python_kind");
  return kind && kind.getValue() == "NoneType";
}

// Walk one front-level subscript slot and record each leaf as either a
// unit-axis insertion (`None`) or a consuming subscript. Front-side
// `hc_front.tuple` slots are unpacked here so the `None` check lands on
// the original constant, not the lowered `hc.const : !hc.undef`. The
// HC values are produced via `lowerValueOperand` so the same value-map
// path used elsewhere applies.
LogicalResult
Lowerer::classifyFrontSubscriptIndex(Value frontIdx, hc_front::SubscriptOp op,
                                     SmallVectorImpl<Value> &residualIndices,
                                     SmallVectorImpl<int64_t> &unitAxes,
                                     size_t &outputPos) {
  if (auto tuple = frontIdx.getDefiningOp<hc_front::TupleOp>()) {
    for (Value element : tuple.getElements()) {
      if (failed(classifyFrontSubscriptIndex(element, op, residualIndices,
                                             unitAxes, outputPos)))
        return failure();
    }
    return success();
  }
  if (isFrontNoneSubscript(frontIdx)) {
    unitAxes.push_back(static_cast<int64_t>(outputPos++));
    return success();
  }
  FailureOr<Value> lowered =
      lowerValueOperand(frontIdx, op.getOperation(), "subscript index");
  if (failed(lowered))
    return failure();
  if (!*lowered) {
    op.emitOpError("subscript index did not lower");
    return failure();
  }
  residualIndices.push_back(*lowered);
  ++outputPos;
  return success();
}

// Generic `hc.buffer_view` lowering for subscripts that didn't match any
// of the dedicated DSL-method folds above.
Value Lowerer::lowerGenericSubscript(hc_front::SubscriptOp op) {
  FailureOr<Value> baseOr =
      lowerValueOperand(op.getBase(), op.getOperation(), "subscript base");
  if (failed(baseOr))
    return nullptr;
  Value base = *baseOr;
  if (!base) {
    op.emitOpError("subscript base did not lower");
    return nullptr;
  }
  SmallVector<Value> residualIndices;
  SmallVector<int64_t> unitAxes;
  size_t outputPos = 0;
  for (Value idx : op.getIndices()) {
    if (failed(classifyFrontSubscriptIndex(idx, op, residualIndices, unitAxes,
                                           outputPos)))
      return nullptr;
  }
  // `hc.buffer_view` accepts `!hc.undef`, buffer, tensor, and vector roots.
  // Type inference later specializes vector roots to element or fragment
  // projections once the index types are known. A non-empty `unit_axes`
  // is the encoded `np.newaxis` set: positions in the OUTPUT rank where
  // size-1 dims are interleaved into the result shape.
  DenseI64ArrayAttr unitAxesAttr = unitAxes.empty()
                                       ? DenseI64ArrayAttr{}
                                       : builder.getDenseI64ArrayAttr(unitAxes);
  return HCBufferViewOp::create(builder, op.getLoc(), undef, base,
                                residualIndices, unitAxesAttr)
      .getResult();
}

Value Lowerer::lowerSubscript(hc_front::SubscriptOp op) {
  // DSL-method `[]` patterns fold into dedicated hc ops when the base is an
  // `hc_front.attr`:
  //   x.shape[N]      -> hc.buffer_dim
  //   group.group_id[N] / local_id[N] / ... -> hc.getitem(hc.tuple(...), N)
  // Property-style (no `()`) access to launch geometry lands here. The
  // call-style form (`wi.local_id()[N]`) is handled by `trySubscriptFolds`
  // so the default type-inference schedule never sees a launch-geo value as
  // a fake buffer_view base.
  SubscriptFoldResult folded = trySubscriptFolds(op);
  if (folded.consumed)
    return folded.value;
  return lowerGenericSubscript(op);
}

//===----------------------------------------------------------------------===//
// Pass scaffolding. The pass runs on the enclosing `builtin.module`, walks
// every `hc_front` top-level callable, and erases it once the parallel
// `hc` callable has been emitted.
//===----------------------------------------------------------------------===//

struct ConvertHCFrontToHCPass
    : public hc_front::impl::ConvertHCFrontToHCBase<ConvertHCFrontToHCPass> {
  using ConvertHCFrontToHCBase::ConvertHCFrontToHCBase;

  // Walk the module once and partition the top-level callables into
  // intrinsics (lowered first; their signatures may be referenced by
  // user kernels/funcs) and user-visible kernel/func ops.
  static void collectFrontCallables(Operation *root,
                                    SmallVectorImpl<Operation *> &intrinsicOps,
                                    SmallVectorImpl<Operation *> &frontOps) {
    for (Region &region : root->getRegions())
      for (Block &block : region)
        for (Operation &op : block) {
          if (isa<hc_front::KernelOp, hc_front::FuncOp>(op))
            frontOps.push_back(&op);
          else if (isa<hc_front::IntrinsicOp>(op))
            intrinsicOps.push_back(&op);
        }
  }

  // When the module declares exactly one `hc_front.kernel`, its launch
  // metadata is the default for any helper that doesn't carry its own
  // (intrinsics, func helpers compiled in the same module). With more
  // than one kernel there's no unique default — pass back an empty
  // attrs bundle.
  static LaunchMetadataAttrs
  pickDefaultLaunchMetadata(ArrayRef<Operation *> frontOps) {
    Operation *singleKernel = nullptr;
    for (Operation *op : frontOps) {
      if (!isa<hc_front::KernelOp>(op))
        continue;
      if (singleKernel)
        return {};
      singleKernel = op;
    }
    if (!singleKernel)
      return {};
    return launchMetadataAttrsFrom(singleKernel);
  }

  // Lower each callable in `ops` via a fresh `Lowerer`, erasing the
  // source op on success and signalling pass failure on the first
  // error. Used for both intrinsic and user-callable batches.
  LogicalResult lowerCallableBatch(OpBuilder &builder, Type undef,
                                   LaunchMetadataAttrs defaultLaunchMetadata,
                                   ArrayRef<Operation *> ops) {
    for (Operation *op : ops) {
      Lowerer lowerer(builder, undef, defaultLaunchMetadata);
      if (failed(lowerer.lowerCallable(op)))
        return failure();
      op->erase();
    }
    return success();
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();
    UndefType undef = UndefType::get(ctx);

    SmallVector<Operation *> intrinsicOps;
    SmallVector<Operation *> frontOps;
    collectFrontCallables(root, intrinsicOps, frontOps);
    LaunchMetadataAttrs defaultLaunchMetadata =
        pickDefaultLaunchMetadata(frontOps);

    OpBuilder builder(ctx);
    if (failed(lowerCallableBatch(builder, undef, defaultLaunchMetadata,
                                  intrinsicOps)) ||
        failed(lowerCallableBatch(builder, undef, defaultLaunchMetadata,
                                  frontOps))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

// `createConvertHCFrontToHCPass()` is emitted by tablegen (friend of
// the impl::ConvertHCFrontToHCBase CRTP). See `Passes.td` — no `let
// constructor`, so the generated factory is the only one.
