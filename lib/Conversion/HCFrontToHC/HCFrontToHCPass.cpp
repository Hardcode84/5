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
// bodies are simulator fallbacks with no compilation meaning; the
// lowered `hc.intrinsic` is a declaration (signature + scope/effects/
// const_kwargs + empty entry block with param args, zero body ops —
// no `hc.assign` either, since there is no body scan downstream for
// them to seed). A consequence worth spelling out: this pass does
// *not* validate the contents of an intrinsic body. Malformed ops
// inside a simulator fallback pass through as-is until the source op
// is erased.
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

std::optional<Type> resolveNumpyDtypeType(MLIRContext *ctx, StringRef name) {
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
Attribute coerceNumpyLiteral(Type targetTy, Attribute src) {
  if (auto ft = dyn_cast<FloatType>(targetTy)) {
    if (auto f = dyn_cast<FloatAttr>(src))
      return FloatAttr::get(ft, f.getValueAsDouble());
    if (auto i = dyn_cast<IntegerAttr>(src))
      return FloatAttr::get(ft, static_cast<double>(i.getInt()));
    return {};
  }
  if (auto it = dyn_cast<IntegerType>(targetTy)) {
    bool isI1 = it.getWidth() == 1;
    auto wrapInteger = [&](int64_t value) {
      APInt bits(64, static_cast<uint64_t>(value), /*isSigned=*/true);
      return IntegerAttr::get(it, bits.sextOrTrunc(it.getWidth()));
    };
    if (auto i = dyn_cast<IntegerAttr>(src)) {
      if (isI1)
        return IntegerAttr::get(it, i.getValue().isZero() ? 0 : 1);
      APInt bits = i.getValue().sextOrTrunc(it.getWidth());
      return IntegerAttr::get(it, bits);
    }
    if (auto f = dyn_cast<FloatAttr>(src)) {
      double v = f.getValueAsDouble();
      if (isI1)
        return IntegerAttr::get(it, v != 0.0 ? 1 : 0);
      if (!std::isfinite(v))
        return {};
      if (v < -0x1.0p63 || v >= 0x1.0p63)
        return {};
      return wrapInteger(static_cast<int64_t>(v));
    }
    return {};
  }
  return {};
}

//===----------------------------------------------------------------------===//
// `group.load(a[row_sl, col_sl], shape=(M, K))` — the WMMA pattern —
// has to land as `hc.load %a[%row_sl, %col_sl] {shape=...}`, not as a
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
Value peelBufferView(Value handle, SmallVectorImpl<Value> &extraIndices) {
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
//===----------------------------------------------------------------------===//

Value emitBinop(OpBuilder &builder, Location loc, StringRef kind, Value lhs,
                Value rhs, Type undef, Operation *sourceOp) {
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

static FailureOr<ExprAttr> subgroupSizeToExprAttr(MLIRContext *ctx,
                                                  Operation *sourceOp,
                                                  IntegerAttr attr) {
  if (!attr)
    return ExprAttr();
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
parseLaunchMetadata(MLIRContext *ctx, Operation *sourceOp,
                    LaunchMetadataAttrs attrs) {
  LaunchMetadata metadata;
  FailureOr<ExprAttr> subgroupSize =
      subgroupSizeToExprAttr(ctx, sourceOp, attrs.subgroupSize);
  if (failed(subgroupSize))
    return failure();
  metadata.subgroupSize = *subgroupSize;
  if (attrs.workShape) {
    FailureOr<ShapeAttr> parsed =
        stringArrayToShape(ctx, sourceOp, attrs.workShape);
    if (failed(parsed))
      return failure();
    metadata.workShape = *parsed;
  }
  if (attrs.groupShape) {
    FailureOr<ShapeAttr> parsed =
        stringArrayToShape(ctx, sourceOp, attrs.groupShape);
    if (failed(parsed))
      return failure();
    metadata.groupShape = *parsed;
  }
  return metadata;
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

static FailureOr<Type> parameterTypeFromDict(
    MLIRContext *ctx, Operation *sourceOp, DictionaryAttr param, Type fallback,
    LaunchMetadataAttrs defaultLaunchMetadata, unsigned paramIndex) {
  auto kind = param.getAs<StringAttr>("kind");
  auto shapeAttr = param.getAs<ArrayAttr>("shape");
  auto name = param.getAs<StringAttr>("name");
  if (!name)
    return sourceOp->emitOpError("`parameters` entry missing `name` key");
  LaunchMetadataAttrs sourceMetadata = launchMetadataAttrsFrom(sourceOp);
  LaunchMetadataAttrs metadataAttrs =
      !sourceMetadata.empty() ? sourceMetadata : defaultLaunchMetadata;
  if (std::optional<StringRef> launchContext =
          getLaunchContextParameterKind(param)) {
    StringRef expected = "";
    if (auto scopedExpected = getScopedLaunchContextParameterKind(sourceOp))
      expected = *scopedExpected;
    if (failed(validateLaunchContextParameter(sourceOp, param, paramIndex,
                                              *launchContext, expected)))
      return failure();
    FailureOr<LaunchMetadata> metadata =
        parseLaunchMetadata(ctx, sourceOp, metadataAttrs);
    if (failed(metadata))
      return failure();
    if (*launchContext == "group")
      return Type(GroupType::get(ctx, metadata->workShape, metadata->groupShape,
                                 metadata->subgroupSize));
    if (*launchContext == "workitem")
      return Type(
          WorkitemType::get(ctx, metadata->groupShape, metadata->subgroupSize));
    if (*launchContext == "subgroup")
      return Type(
          SubgroupType::get(ctx, metadata->groupShape, metadata->subgroupSize));
    return sourceOp->emitOpError("unknown launch-context parameter kind '")
           << *launchContext << "'";
  }
  if (auto scopedExpected = getScopedLaunchContextParameterKind(sourceOp);
      scopedExpected && paramIndex == 0)
    return sourceOp->emitOpError("first scoped helper parameter must be "
                                 "marked as a ")
           << *scopedExpected << " launch context";
  if (name.getValue() == "group" &&
      !getScopedLaunchContextParameterKind(sourceOp)) {
    FailureOr<LaunchMetadata> metadata =
        parseLaunchMetadata(ctx, sourceOp, metadataAttrs);
    if (failed(metadata))
      return failure();
    return Type(GroupType::get(ctx, metadata->workShape, metadata->groupShape,
                               metadata->subgroupSize));
  }
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

  if (!shapeAttr)
    return sourceOp->emitOpError("buffer parameter '")
           << name.getValue() << "' is missing `shape` metadata";

  Type elementType = fallback;
  if (auto dtype = param.getAs<StringAttr>("dtype")) {
    std::optional<Type> resolved = resolveNumpyDtypeType(ctx, dtype.getValue());
    if (!resolved)
      return sourceOp->emitOpError("buffer parameter '")
             << name.getValue() << "' has unsupported dtype '"
             << dtype.getValue() << "'";
    elementType = *resolved;
  }

  FailureOr<ShapeAttr> shape = stringArrayToShape(ctx, sourceOp, shapeAttr);
  if (failed(shape))
    return failure();
  return Type(BufferType::get(ctx, elementType, *shape));
}

//===----------------------------------------------------------------------===//
// Lowerer. Instantiated once per top-level `hc_front` callable; walks the
// body, threads the scope map through nested regions, and emits the
// corresponding `hc` ops via the shared `builder`. Source ops are erased
// after each top-level op finishes lowering.
//===----------------------------------------------------------------------===//

class Lowerer {
public:
  Lowerer(OpBuilder &builder, Type undef,
          LaunchMetadataAttrs defaultLaunchMetadata)
      : builder(builder), undef(undef),
        defaultLaunchMetadata(defaultLaunchMetadata) {}

  LogicalResult lowerCallable(Operation *frontOp);

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

  LogicalResult lowerOp(Operation *op);

  FailureOr<Value> lowerValueOperand(Value v, Operation *consumer,
                                     StringRef role = "operand");
  FailureOr<SmallVector<Value>> lowerValueOperands(ValueRange vs,
                                                   Operation *consumer,
                                                   StringRef role = "operand");
  FailureOr<SmallVector<Value>> expandTupleOperand(Value v, Operation *consumer,
                                                   StringRef role);
  FailureOr<SmallVector<Value>> lowerReturnValues(hc_front::ReturnOp op);

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
  // Flattens a `hc_front.inlined_region` into the caller's current `hc`
  // insertion block. The region is a pure name-scope boundary stamped
  // by `-hc-front-inline`; consuming it here keeps the `hc` output
  // free of `hc_front` artifacts. Alpha-renames every local name in
  // the cloned body with a per-site prefix, emits `hc.assign` for each
  // parameter binding, walks the body through `lowerOp`, and intercepts
  // `hc_front.return` to wire the region's results into `valueMap`.
  LogicalResult lowerInlinedRegion(hc_front::InlinedRegionOp op);
  // FailureOr so a `builtin` call that is cleanly consumed by its parent
  // (`range(...)` inside `hc_front.for`) can be distinguished from a real
  // error. `FailureOr<Value>` reads as: success+Value / success+null /
  // failure.
  FailureOr<Value> lowerCall(hc_front::CallOp op);
  Value lowerSubscript(hc_front::SubscriptOp op);

  // Populates `entry`'s block arguments and returns the matching
  // `FunctionType`. Buffer parameters are seeded from their metadata;
  // everything else starts as `!hc.undef`. No `hc.assign` is emitted here;
  // param-name publishing happens in `emitParameterAssigns` once the
  // enclosing op has attached `entry` to its body.
  FailureOr<FunctionType> materializeParameters(ArrayAttr params, Block &entry,
                                                Operation *sourceOp,
                                                bool returnsValue);

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

  // Call-site special-cased kwargs get picked out of the argument list
  // before operands are lowered to real `hc` ops, so the callee can see a
  // flat positional arg list plus a {name: attr} map.
  struct CallArgs {
    SmallVector<Value> positional;
    llvm::StringMap<Attribute> kwattrs;
    llvm::StringMap<Value> kwvalues;
  };
  FailureOr<CallArgs> collectCallArgs(hc_front::CallOp op);

  FailureOr<Value> lowerNumpyDtypeCall(hc_front::CallOp call,
                                       const RefInfo &ref,
                                       const CallArgs &args);
  FailureOr<Value> lowerUnaryBaseMethod(hc_front::CallOp call, StringRef method,
                                        Value base, const CallArgs &args);
  FailureOr<Value> lowerMemOp(hc_front::CallOp call, StringRef method,
                              const CallArgs &args);
  Value tryLowerLaunchGeoCall(hc_front::CallOp call, StringRef method,
                              Value base, const CallArgs &args);

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
  collectStaticLaunchGeometryRanks(frontOp);

  // All three callable kinds share the same skeleton: build a fresh
  // entry block with one block arg per param, hand (entry, fnType) to
  // the per-kind `build` lambda which creates the hc op + attaches the
  // block, then (if `build` asks for it) emit a leading `hc.assign` per
  // param and lower the hc_front body region into that entry block.
  // `build` returning a null `Region *` on success means "skip the walk"
  // — the intrinsic arm uses it to land a declaration-only op with an
  // empty (no-assigns, no-body) entry block; see file banner.
  //
  // Block ownership: on `materializeParameters` failure we delete
  // `entry` here; on success `build` is contractually attach-then-erase
  // — attaches `entry` to the hc op first, so any later failure can
  // `hcOp->erase()` and take the block with it.
  auto runBody =
      [&](ArrayAttr runParams, bool returnsValue,
          llvm::function_ref<FailureOr<Region *>(Block *, FunctionType)> build)
      -> LogicalResult {
    Block *entry = new Block();
    auto fnType =
        materializeParameters(runParams, *entry, frontOp, returnsValue);
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
    // `emitParameterAssigns` leaves the builder just past the last
    // parameter `hc.assign`, which is where we want `lowerRegion` to
    // start appending the body's converted ops.
    return lowerRegion(**bodyRegion);
  };

  if (auto kernel = dyn_cast<hc_front::KernelOp>(frontOp)) {
    return runBody(
        params, /*returnsValue=*/false,
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
            workRank = static_cast<unsigned>((*shape).getDims().size());
          }
          if (auto gs = frontOp->getAttrOfType<ArrayAttr>("group_shape")) {
            auto shape = stringArrayToShape(ctx, frontOp, gs);
            if (failed(shape)) {
              hcKernel->erase();
              return failure();
            }
            hcKernel.setGroupShapeAttr(*shape);
            groupRank = static_cast<unsigned>((*shape).getDims().size());
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
        params, /*returnsValue=*/true,
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
    auto constKwargsAttr = frontOp->getAttrOfType<ArrayAttr>("const_kwargs");
    if (constKwargsAttr) {
      for (auto [idx, kw] : llvm::enumerate(constKwargsAttr)) {
        if (!isa<StringAttr>(kw))
          return frontOp->emitOpError("`const_kwargs` entry at index ")
                 << idx << " must be a StringAttr, got " << kw;
      }
    }

    FailureOr<ArrayAttr> parameterNames =
        parameterNamesFromDicts(params, frontOp);
    if (failed(parameterNames))
      return failure();
    FailureOr<ArrayAttr> keywordOnlyParameters =
        keywordOnlyParametersFromDicts(params, frontOp);
    if (failed(keywordOnlyParameters))
      return failure();
    llvm::SmallDenseSet<StringRef> keywordOnlySet;
    for (Attribute kw : *keywordOnlyParameters)
      keywordOnlySet.insert(cast<StringAttr>(kw).getValue());
    if (constKwargsAttr) {
      for (Attribute kw : constKwargsAttr) {
        StringRef name = cast<StringAttr>(kw).getValue();
        if (!keywordOnlySet.contains(name))
          return frontOp->emitOpError("const kwarg '")
                 << name << "' must be declared keyword-only";
      }
    }

    // `hc.intrinsic` owns the const-kwarg filtering rule: its
    // `function_type` is the runtime operand signature, while
    // `parameters` keeps the full declared order for call-site kwarg
    // binding. Intrinsics keep runtime operand types erased here because
    // target-specific lowering validates and specializes their operands.
    FunctionType fnType = getIntrinsicOperandFunctionType(
        *parameterNames, constKwargsAttr, TypeRange{undef}, undef);
    ArrayAttr parameterNamesAttr = *parameterNames;
    ArrayAttr keywordOnlyParametersAttr = *keywordOnlyParameters;
    Block *entry = new Block();
    for (Type input : fnType.getInputs())
      entry->addArgument(input, loc);

    auto hcIntr = HCIntrinsicOp::create(
        builder, loc, StringAttr::get(ctx, intr.getName()),
        TypeAttr::get(fnType), ScopeAttr::get(ctx, scopeAttr.getValue()),
        /*effects=*/EffectClassAttr(), /*const_kwargs=*/ArrayAttr(),
        /*parameters=*/parameterNamesAttr,
        /*keyword_only=*/keywordOnlyParametersAttr);
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

  return frontOp->emitOpError("unexpected top-level hc_front op");
}

FailureOr<FunctionType> Lowerer::materializeParameters(ArrayAttr params,
                                                       Block &entry,
                                                       Operation *sourceOp,
                                                       bool returnsValue) {
  MLIRContext *ctx = sourceOp->getContext();
  if (auto scopedExpected = getScopedLaunchContextParameterKind(sourceOp)) {
    for (auto [idx, param] : llvm::enumerate(params)) {
      auto dict = dyn_cast<DictionaryAttr>(param);
      if (!dict)
        continue;
      std::optional<StringRef> launchContext =
          getLaunchContextParameterKind(dict);
      if (!launchContext)
        continue;
      if (failed(validateLaunchContextParameter(
              sourceOp, dict, idx, *launchContext, *scopedExpected)))
        return failure();
      break;
    }
  }
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
        parameterTypeFromDict(ctx, sourceOp, dict, undef, defaultLaunchMetadata,
                              static_cast<unsigned>(inputs.size()));
    if (failed(paramType))
      return failure();
    if (auto groupType = dyn_cast<GroupType>(*paramType)) {
      launchWorkShape = groupType.getWorkShape();
      launchGroupShape = groupType.getGroupShape();
      launchSubgroupSize = groupType.getSubgroupSize();
      if (launchWorkShape)
        workRank = static_cast<unsigned>(launchWorkShape.getDims().size());
      if (launchGroupShape)
        groupRank = static_cast<unsigned>(launchGroupShape.getDims().size());
    } else if (auto workitemType = dyn_cast<WorkitemType>(*paramType)) {
      launchGroupShape = workitemType.getGroupShape();
      launchSubgroupSize = workitemType.getSubgroupSize();
      if (launchGroupShape)
        groupRank = static_cast<unsigned>(launchGroupShape.getDims().size());
    } else if (auto subgroupType = dyn_cast<SubgroupType>(*paramType)) {
      launchGroupShape = subgroupType.getGroupShape();
      launchSubgroupSize = subgroupType.getSubgroupSize();
      if (launchGroupShape)
        groupRank = static_cast<unsigned>(launchGroupShape.getDims().size());
    }
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
    FailureOr<Value> v = lowerAttr(attr);
    if (failed(v))
      return failure();
    valueMap[attr.getResult()] = *v;
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
  if (auto ir = dyn_cast<hc_front::InlinedRegionOp>(op))
    return lowerInlinedRegion(ir);
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
    FailureOr<SmallVector<Value>> elts =
        lowerValueOperands(t.getElements(), op, "tuple element");
    if (failed(elts))
      return failure();
    if (llvm::any_of(*elts, [](Value v) { return !v; }))
      return t.emitOpError("tuple element did not lower to an hc value");
    SmallVector<Type> elementTypes;
    elementTypes.reserve(elts->size());
    for (Value elt : *elts)
      elementTypes.push_back(elt.getType());
    Type tupleType = TupleType::get(op->getContext(), elementTypes);
    valueMap[t.getResult()] =
        HCTupleOp::create(builder, op->getLoc(), tupleType, *elts);
    return success();
  }
  if (auto k = dyn_cast<hc_front::KeywordOp>(op)) {
    FailureOr<Value> v = lowerValueOperand(k.getValue(), op, "keyword value");
    if (failed(v))
      return failure();
    keywordInfo[k.getResult()] = {k.getName(), *v};
    valueMap[k.getResult()] = Value();
    return success();
  }

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

FailureOr<Value> Lowerer::lowerAttr(hc_front::AttrOp op) {
  // Attribute chains like `group.load` or `buf.shape` are folded into
  // the parent call or subscript and don't produce a standalone SSA
  // value at this layer. The one exception is `numpy_dtype_type`: the
  // attr itself denotes a type, which both `x.astype(np.<dtype>)` (arg
  // position) and `np.<dtype>(0)` (callee position) want to observe as
  // an `hc.const` carrying a `TypeAttr`. Materializing it eagerly here
  // keeps `collectCallArgs` honest (no null-arg rejection) and lets
  // `lowerDslMethodCall` reuse the same value when a numpy dtype is
  // used as a value constructor.
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
  return Value();
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

LogicalResult Lowerer::lowerAssign(hc_front::AssignOp op) {
  Value rhs = op.getValue();
  Operation *target = op.getTarget().getDefiningOp();
  MLIRContext *ctx = op.getContext();
  auto nameAttr = [&](StringRef n) { return StringAttr::get(ctx, n); };

  if (auto tn = dyn_cast_if_present<hc_front::TargetNameOp>(target)) {
    FailureOr<Value> valueOr =
        lowerValueOperand(rhs, op.getOperation(), "assignment rhs");
    if (failed(valueOr))
      return failure();
    Value value = *valueOr;
    if (!value)
      return op.emitOpError("rhs for '") << tn.getName()
                                         << "' did not lower to an hc value; "
                                            "ref classification may be off";
    HCAssignOp::create(builder, op.getLoc(), nameAttr(tn.getName()), value);
    return success();
  }

  if (auto tt = dyn_cast_if_present<hc_front::TargetTupleOp>(target)) {
    // Multi-assign: `a, b = <rhs>`. First-class tuple SSA values are
    // destructured with `hc.getitem`; truly multi-result front ops can still
    // distribute their individual results. Arity-1 unpack also uses getitem so
    // `a, = scalar` does not silently become scalar assignment.
    size_t arity = tt.getElements().size();
    SmallVector<Value> sources;
    FailureOr<Value> valueOr =
        lowerValueOperand(rhs, op.getOperation(), "tuple-unpack rhs");
    if (failed(valueOr))
      return failure();
    Value value = *valueOr;
    Operation *rhsOp = value ? value.getDefiningOp() : nullptr;
    if (rhsOp && rhsOp->getNumResults() == arity && arity != 1) {
      sources.assign(rhsOp->getResults().begin(), rhsOp->getResults().end());
    } else if (value) {
      for (size_t i = 0; i < arity; ++i) {
        auto index =
            HCConstOp::create(builder, op.getLoc(), undef,
                              IntegerAttr::get(IntegerType::get(ctx, 64),
                                               static_cast<int64_t>(i)));
        sources.push_back(HCGetItemOp::create(builder, op.getLoc(), undef,
                                              value, index.getResult()));
      }
    } else {
      return op.emitOpError("tuple-unpack rhs did not lower to an hc value");
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
      HCAssignOp::create(builder, op.getLoc(), nameAttr(tn.getName()), src);
    }
    return success();
  }

  if (auto ts = dyn_cast_if_present<hc_front::TargetSubscriptOp>(target)) {
    FailureOr<Value> valueOr =
        lowerValueOperand(rhs, op.getOperation(), "store source");
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
  // `hc_front.for` declares all three sub-regions as `SizedRegion<1>`, so
  // multi-block bodies cannot reach this pass — but keep a runtime check
  // so a verifier-bypass doesn't silently walk only the first block under
  // NDEBUG.
  if (!iter.hasOneBlock())
    return op.emitOpError(
        "for-iter must be single-block (dialect verifier bypassed?)");

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
  RefInfo calleeRef = RefInfo::get(calleeName);
  if (!calleeName || calleeRef.getKind() != "builtin" ||
      calleeRef.getString("builtin") != "range") {
    return op.emitOpError("for-iter must be `range(...)`");
  }

  // Python-style `range(stop)` / `range(start, stop)` / `range(start, stop,
  // step)`: pad the missing parts with the canonical defaults so the
  // `hc.for_range` op always sees three operands.
  FailureOr<SmallVector<Value>> rangeArgsOr = lowerValueOperands(
      iterCall.getArguments(), op.getOperation(), "range argument");
  if (failed(rangeArgsOr))
    return failure();
  SmallVector<Value> rangeArgs = std::move(*rangeArgsOr);
  if (llvm::any_of(rangeArgs, [](Value v) { return !v; }))
    return iterCall.emitOpError("range argument did not lower to an hc value");
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
static void alphaRenameInlinedBody(Region &body, StringRef prefix) {
  SmallVector<Region *> worklist{&body};
  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    for (Block &block : *region) {
      for (Operation &op : block) {
        if (auto name = dyn_cast<hc_front::NameOp>(&op)) {
          if (auto ref = name->getAttrOfType<DictionaryAttr>("ref")) {
            if (auto kind = ref.getAs<StringAttr>("kind")) {
              StringRef k = kind.getValue();
              if (k == "param" || k == "local" || k == "iv")
                name.setName((prefix + name.getName()).str());
            }
          }
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
}

LogicalResult Lowerer::lowerInlinedRegion(hc_front::InlinedRegionOp op) {
  MLIRContext *ctx = op.getContext();
  Region &body = op.getBody();
  if (body.empty() || !body.hasOneBlock())
    return op.emitOpError("inlined_region body must be single-block");

  auto params = op->getAttrOfType<ArrayAttr>("parameters");
  if (!params || params.size() != op.getArguments().size()) {
    return op.emitOpError(
               "inlined_region parameter/argument arity mismatch (params=")
           << (params ? params.size() : 0)
           << ", args=" << op.getArguments().size() << ")";
  }

  // Per-site prefix. `inlineSiteCounter` is module-wide, so two sites
  // of the same helper in the same caller still get distinct prefixes.
  std::string prefix =
      ("__inl_" + op.getCallee() + "_" + Twine(inlineSiteCounter) + "_").str();
  ++inlineSiteCounter;

  alphaRenameInlinedBody(body, prefix);

  // Bind params at the current caller insertion point: `hc.assign
  // @<renamed_param> = <lowered_arg>`. The body's `hc_front.name`
  // references to the param now read the prefixed name, which the
  // promotion pass folds against these assigns.
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
    if (!loweredArg) {
      return op.emitOpError("inline argument for param `")
             << nameAttr.getValue() << "' did not lower to an hc value";
    }
    std::string renamed = (prefix + nameAttr.getValue()).str();
    HCAssignOp::create(builder, op.getLoc(), StringAttr::get(ctx, renamed),
                       loweredArg);
  }

  // Walk the body ops. `hc_front.return` is the one we can't feed
  // through `lowerOp`: the generic path would emit `hc.return` into
  // the caller, which is wrong — the return's operands are instead
  // the region's *result values* and must be funneled into `valueMap` so
  // downstream consumers (`hc_front.assign`, etc.) resolve through the normal
  // paths. Tuple returns are first-class values; only explicit multi-operand
  // returns create multiple region results.
  SmallVector<Value> resultValues;
  bool sawReturn = false;
  for (Operation &child : llvm::make_early_inc_range(body.front())) {
    if (auto r = dyn_cast<hc_front::ReturnOp>(&child)) {
      if (sawReturn)
        return op.emitOpError("inlined_region body has multiple returns");
      sawReturn = true;
      resultValues.clear();
      resultValues.reserve(r.getValues().size());
      for (Value v : r.getValues()) {
        FailureOr<Value> loweredOr = lowerValueOperand(
            v, r.getOperation(), "inlined_region return operand");
        if (failed(loweredOr))
          return failure();
        Value lowered = *loweredOr;
        if (!lowered) {
          return r.emitOpError("inlined_region return operand did not lower");
        }
        resultValues.push_back(lowered);
      }
      continue;
    }
    if (failed(lowerOp(&child)))
      return failure();
  }
  if (!sawReturn)
    return op.emitOpError("inlined_region body has no return");

  // Map region results. `hc_front.call` has one SSA result, so the canonical
  // lowering for multiple explicit return operands is one tuple value at result
  // #0. Additional region results keep hand-written direct uses well-defined;
  // the front dialect verifier pins their arity to the body return.
  unsigned nResults = op.getNumResults();
  if (nResults == 0) {
    if (!resultValues.empty()) {
      return op.emitOpError(
          "inlined_region declares no results but body returns values");
    }
    return success();
  }
  if (resultValues.size() != nResults) {
    return op.emitOpError(
               "inlined_region result arity mismatch: region declares ")
           << nResults << ", body returns " << resultValues.size();
  }
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

template <typename HCRegionOpT, typename FrontRegionOpT>
LogicalResult Lowerer::lowerCapturingRegion(FrontRegionOpT op) {
  // `hc.{workitem,subgroup}_region` match `hc_front` 1:1 (captures +
  // body). Nested-def folding is a later pass; if the front op declares
  // parameters we still add them as block args, plus one leading
  // `hc.assign "<p>", %arg` per param so the body's name lookups
  // resolve via the promotion pass. The `hc` op carries only a
  // captures list (no formal params), matching ODS.
  //
  bool sourceEmpty = op.getBody().empty();
  if (!sourceEmpty && !op.getBody().hasOneBlock())
    return op.emitOpError("hc_front nested region must be single-block");

  // Folded `return inner()` regions lower to ordinary SSA control flow:
  // yield from the nested scope, then return from the enclosing callable.
  bool isTailReturnRegion = op.getTailReturnAttr() != nullptr;
  SmallVector<Type> resultTypes;
  if (isTailReturnRegion) {
    if (sourceEmpty)
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

    resultTypes.assign(soleReturn.getValues().size(), undef);
  }

  auto newOp = HCRegionOpT::create(builder, op.getLoc(), resultTypes,
                                   op.getCapturesAttr());
  Block *body = new Block();
  auto params = op->template getAttrOfType<ArrayAttr>("parameters");
  StringRef expectedLaunchContext =
      isa<HCWorkitemRegionOp>(newOp.getOperation()) ? StringRef("workitem")
                                                    : StringRef("subgroup");
  // Region parameters must carry the same explicit launch-context marker as
  // scoped helper functions; remaining source parameters start erased.
  auto regionParamType = [&](DictionaryAttr dict,
                             unsigned index) -> FailureOr<Type> {
    if (std::optional<StringRef> launchContext =
            getLaunchContextParameterKind(dict)) {
      if (failed(validateLaunchContextParameter(op.getOperation(), dict, index,
                                                *launchContext,
                                                expectedLaunchContext)))
        return failure();
    } else if (index == 0) {
      return op.emitOpError(
                 "first nested region parameter must be marked as a ")
             << expectedLaunchContext << " launch context";
    } else {
      return Type(undef);
    }
    if (isa<HCWorkitemRegionOp>(newOp.getOperation()))
      return Type(WorkitemType::get(op->getContext(), launchGroupShape,
                                    launchSubgroupSize));
    if (isa<HCSubgroupRegionOp>(newOp.getOperation()))
      return Type(SubgroupType::get(op->getContext(), launchGroupShape,
                                    launchSubgroupSize));
    return Type(undef);
  };
  SmallVector<StringAttr> paramNames;
  if (params) {
    paramNames.reserve(params.size());
    for (Attribute p : params) {
      auto dict = dyn_cast<DictionaryAttr>(p);
      if (!dict)
        return op.emitOpError("invalid parameters entry");
      auto name = dict.template getAs<StringAttr>("name");
      if (!name)
        return op.emitOpError("parameters entry missing `name`");
      FailureOr<Type> paramType = regionParamType(dict, paramNames.size());
      if (failed(paramType))
        return failure();
      body->addArgument(*paramType, op.getLoc());
      paramNames.push_back(name);
    }
  }
  newOp.getBody().push_back(body);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    for (auto [idx, name] : llvm::enumerate(paramNames))
      HCAssignOp::create(builder, op.getLoc(), name, body->getArgument(idx));

    if (!sourceEmpty) {
      SmallVector<Value> returnValues;
      for (Operation &child :
           llvm::make_early_inc_range(op.getBody().front())) {
        if (isTailReturnRegion) {
          if (auto retOp = dyn_cast<hc_front::ReturnOp>(&child)) {
            FailureOr<SmallVector<Value>> loweredValues =
                lowerReturnValues(retOp);
            if (failed(loweredValues))
              return failure();
            returnValues = std::move(*loweredValues);
            if (returnValues.size() != newOp->getNumResults())
              return retOp.emitOpError("return arity mismatch for tail-return "
                                       "region");
            HCYieldOp::create(builder, retOp.getLoc(), returnValues);
            continue;
          }
        }
        if (failed(lowerOp(&child)))
          return failure();
      }
    }
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

FailureOr<Lowerer::CallArgs> Lowerer::collectCallArgs(hc_front::CallOp op) {
  CallArgs args;
  for (Value arg : op.getArguments()) {
    auto kwIt = keywordInfo.find(arg);
    if (kwIt != keywordInfo.end()) {
      // `shape = (M, N)` folds to a shape attr at the call site and
      // lands in `kwattrs`. Everything else lands in `kwvalues` as a
      // candidate SSA operand for `lowerCall` to match against the
      // callee's declared `parameters` (excluding names in the
      // `const_kwargs` whitelist, which are promoted back to
      // call-site attributes).
      StringRef kwName = kwIt->second.name;
      Value kwVal = kwIt->second.loweredValue;
      if (kwName == "shape") {
        // The tuple-of-constants -> `#hc.shape<...>` fold. Derive the element
        // list from the lowered `hc.tuple`; if it is not a static tuple of
        // integer/string constants, fall back to storing the tuple value so a
        // later lowering can still see it.
        if (auto tuple = kwVal ? kwVal.getDefiningOp<HCTupleOp>() : nullptr) {
          SmallVector<Attribute> dims;
          dims.reserve(tuple.getElements().size());
          auto &store =
              op->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
          // If every element is an `hc.const` carrying an integer, we can
          // materialize a shape attribute. String-valued constants are
          // also accepted (they already name a symbol).
          bool ok = true;
          for (Value e : tuple.getElements()) {
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

  FailureOr<CallArgs> argsOr = collectCallArgs(op);
  if (failed(argsOr))
    return failure();
  CallArgs &args = *argsOr;

  if (kind == "callee" || kind == "intrinsic") {
    StringRef callee = ref.getString("callee");
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

    // The lowered `hc.intrinsic` symbol carries both the full declared
    // parameter order and the const-kwarg whitelist. Intrinsic
    // declarations are materialized before caller bodies are lowered, so
    // this lookup is independent of source order.
    HCIntrinsicOp intrDecl =
        SymbolTable::lookupNearestSymbolFrom<HCIntrinsicOp>(op.getOperation(),
                                                            symRef);
    if (!intrDecl) {
      op.emitOpError("intrinsic '@")
          << callee << "' does not resolve to a lowered hc.intrinsic";
      return failure();
    }
    ArrayAttr declaredParameters = intrDecl.getParametersAttr();
    ArrayAttr constKwargsAttr = intrDecl.getConstKwargsAttr();
    ArrayAttr keywordOnlyAttr = intrDecl.getKeywordOnlyAttr();
    llvm::SmallDenseSet<StringRef> constKwargSet;
    if (constKwargsAttr) {
      for (Attribute kw : constKwargsAttr)
        if (auto s = dyn_cast<StringAttr>(kw))
          constKwargSet.insert(s.getValue());
    }
    llvm::SmallDenseSet<StringRef> keywordOnlySet;
    if (keywordOnlyAttr) {
      for (Attribute kw : keywordOnlyAttr)
        if (auto s = dyn_cast<StringAttr>(kw))
          keywordOnlySet.insert(s.getValue());
    }

    // Walk the callee's declared parameter list to anchor operand order:
    // positionals bind only the prefix before any keyword-only marker,
    // keyword operands bind only keyword-only slots, and const_kwargs are
    // skipped (they land as call-site attributes below). Matching by
    // declared-parameter index rather than `kwvalues` iteration keeps the
    // order deterministic across hash layouts.
    SmallVector<Value> operands(args.positional.begin(), args.positional.end());
    llvm::SmallDenseSet<StringRef> consumedKwargs;
    if (declaredParameters) {
      ArrayAttr params = declaredParameters;
      if (operands.size() > params.size()) {
        op.emitOpError("intrinsic '@")
            << callee << "' declares " << params.size()
            << " parameter(s), call site supplies " << operands.size()
            << " positional";
        return failure();
      }
      for (unsigned i = 0, e = operands.size(); i < e; ++i) {
        auto nameAttr = dyn_cast<StringAttr>(params[i]);
        if (!nameAttr) {
          op.emitOpError("intrinsic '@")
              << callee << "' has malformed `parameters` entry at index " << i
              << ": expected StringAttr, got " << params[i];
          return failure();
        }
        if (keywordOnlySet.contains(nameAttr.getValue())) {
          op.emitOpError("intrinsic '@")
              << callee << "' parameter '" << nameAttr.getValue()
              << "' is keyword-only and cannot be passed positionally";
          return failure();
        }
      }
      for (unsigned i = operands.size(), e = params.size(); i < e; ++i) {
        auto nameAttr = dyn_cast<StringAttr>(params[i]);
        if (!nameAttr) {
          op.emitOpError("intrinsic '@")
              << callee << "' has malformed `parameters` entry at index " << i
              << ": expected StringAttr, got " << params[i];
          return failure();
        }
        StringRef pname = nameAttr.getValue();
        if (constKwargSet.contains(pname))
          continue;
        // Intrinsic IR reserves `hc_front.keyword` operands for Python
        // keyword-only parameters. Positional-or-keyword spelling would make
        // hand-authored IR depend on source argument order rather than the
        // callee's declared ABI.
        if (keywordOnlyAttr && !keywordOnlySet.contains(pname)) {
          if (args.kwvalues.contains(pname)) {
            op.emitOpError("intrinsic '@")
                << callee << "' parameter '" << pname
                << "' is positional and cannot be passed as a keyword";
            return failure();
          }
          op.emitOpError("intrinsic '@")
              << callee << "' missing required positional argument '" << pname
              << "'";
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
      }
    } else if (!args.kwvalues.empty()) {
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

    // Kwargs the declared walk didn't consume and that aren't on the
    // const whitelist are unknown at this call site. Split the
    // diagnostics along the axis they arrive on so triage doesn't have
    // to cross-reference `collectCallArgs`.
    for (auto &kv : args.kwvalues) {
      StringRef name = kv.first();
      if (consumedKwargs.contains(name) || constKwargSet.contains(name))
        continue;
      op.emitOpError("intrinsic '@")
          << callee << "' called with unknown keyword argument '" << name
          << "'";
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

    auto call = HCCallIntrinsicOp::create(builder, op.getLoc(),
                                          TypeRange{undef}, symRef, operands);
    if (constKwargsAttr) {
      // Promote declared const kwargs to call-site attributes: folded
      // shape attr if present, else the producing `hc.const`'s payload.
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
    // `-hc-front-inline` is responsible for consuming every inline call
    // before we run. If one survives to here, either the pipeline was
    // ordered wrong or the pass missed a site — in both cases the
    // operator wants a loud, located diagnostic, not a silent
    // placeholder.
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
    // there is no callable for us to dispatch against. Parallel
    // ordering diagnostic to the `inline` case above.
    op.emitOpError(
        "`ref.kind = \"local\"` call survived to conversion; run "
        "`-hc-front-fold-region-defs` before `-convert-hc-front-to-hc`");
    return failure();
  }
  op.emitOpError("unsupported callee ref.kind '") << kind << "'";
  return failure();
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

  if (ref.getKind() == "numpy_dtype_type")
    return lowerNumpyDtypeCall(call, ref, args);

  FailureOr<Value> baseOr =
      lowerValueOperand(attr.getBase(), call.getOperation(), "method base");
  if (failed(baseOr))
    return failure();
  Value base = *baseOr;

  if (method == "vec" || method == "with_inactive" || method == "astype")
    return lowerUnaryBaseMethod(call, method, base, args);
  if (method == "load" || method == "vload" || method == "store" ||
      method == "vzeros")
    return lowerMemOp(call, method, args);
  if (Value lowered = tryLowerLaunchGeoCall(call, method, base, args))
    return {lowered};

  call.emitOpError("unsupported dsl_method '") << method << "'";
  return failure();
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
    Value inactive = valIt->second;
    if (!inactive) {
      call.emitOpError("with_inactive value did not lower to an hc value");
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
  llvm_unreachable("unknown unary-base DSL method");
}

FailureOr<Value> Lowerer::lowerMemOp(hc_front::CallOp call, StringRef method,
                                     const CallArgs &args) {
  // Buffer/tensor load, vload, store: the frontend passes a (possibly
  // pre-sliced) handle as the first positional, remaining positionals go
  // to the indices list, and an optional `shape=` kwarg lands as the
  // attribute. We rely on `collectCallArgs`' shape-kwarg fold so no manual
  // tuple walking is needed here. `peelBufferView` at file scope folds
  // the pre-subscripted `hc.buffer_view` into the op's own index list;
  // see its banner for the single-level rationale.
  //
  // Chained-subscript guard: if after one peel the handle is *still* a
  // `hc.buffer_view`, the user wrote `a[i][j]`-style nested subscripts.
  // That lowers to two distinct buffer_views whose index lists can't be
  // safely spliced (the outer slice re-indexes the already-reduced view,
  // not the original buffer's next axis). Diagnose with the rewrite
  // suggestion rather than emit wrong IR or let the rank verifier
  // complain about an index count the user didn't write.
  auto rejectNestedView = [&](Value handle) -> LogicalResult {
    if (!handle.getDefiningOp<HCBufferViewOp>())
      return success();
    call.emitOpError("chained subscript into `")
        << method
        << "` is not supported; use a single tuple subscript instead "
           "(e.g. `group."
        << method << "(a[i, j], ...)` rather than `a[i][j]`)";
    return failure();
  };

  if (method == "load" || method == "vload") {
    if (args.positional.empty()) {
      call.emitOpError("`") << method << "` expects a buffer/tensor argument";
      return failure();
    }
    Value src = args.positional.front();
    SmallVector<Value> indices(args.positional.begin() + 1,
                               args.positional.end());
    src = peelBufferView(src, indices);
    if (failed(rejectNestedView(src)))
      return failure();
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
    dest = peelBufferView(dest, indices);
    if (failed(rejectNestedView(dest)))
      return failure();
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
  llvm_unreachable("unknown memory DSL method");
}

Value Lowerer::tryLowerLaunchGeoCall(hc_front::CallOp call, StringRef method,
                                     Value base, const CallArgs &args) {
  // `group.{launch_geo}()` with no args is the call form (`wi.local_id()`).
  // Emit one tuple-valued launch-geo query; an enclosing subscript lowers to
  // `hc.getitem`. The property-style `group.launch_geo[N]` is handled in
  // `lowerSubscript` — both go through `tryEmitLaunchGeo` so the supported
  // method set stays in one place.
  if (call.getArguments().empty() && args.kwvalues.empty() &&
      args.kwattrs.empty()) {
    if (base) {
      std::optional<LaunchGeoMethodInfo> methodInfo =
          classifyLaunchGeoMethod(method);
      if (!methodInfo)
        return {};
      std::optional<unsigned> requiredRank =
          getStaticLaunchGeometryRank(call.getResult(), method);
      if (Value v =
              tryEmitLaunchGeo(*methodInfo, base, call.getLoc(), requiredRank))
        return v;
    }
  }
  return {};
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

void Lowerer::collectStaticLaunchGeometryRanks(Operation *frontOp) {
  auto record = [&](Value source, StringRef method, unsigned rank) {
    unsigned &current = staticLaunchGeoRanks[source][method];
    current = std::max(current, rank);
  };

  frontOp->walk([&](hc_front::SubscriptOp op) {
    if (op.getIndices().size() != 1)
      return;
    Value index = op.getIndices().front();

    if (auto attr = dyn_cast_if_present<hc_front::AttrOp>(
            op.getBase().getDefiningOp())) {
      StringRef method = attr.getName();
      std::optional<LaunchGeoMethodInfo> methodInfo =
          classifyLaunchGeoMethod(method);
      if (!methodInfo || methodInfo->isScalar())
        return;
      std::optional<unsigned> rank = staticLaunchGeoRequiredRank(index);
      if (!rank)
        return;
      record(attr.getBase(), method, *rank);
      return;
    }

    auto call =
        dyn_cast_if_present<hc_front::CallOp>(op.getBase().getDefiningOp());
    if (!call || !call.getArguments().empty())
      return;
    auto attr =
        dyn_cast_if_present<hc_front::AttrOp>(call.getCallee().getDefiningOp());
    if (!attr)
      return;
    StringRef method = attr.getName();
    std::optional<LaunchGeoMethodInfo> methodInfo =
        classifyLaunchGeoMethod(method);
    if (!methodInfo || methodInfo->isScalar())
      return;

    // A call-style launch-geo tuple can also escape as a first-class value. In
    // that case preserve the conservative cap-sized fallback; only pure static
    // getitem use-sites get the tightened rank.
    unsigned rank = 0;
    for (Operation *user : call.getResult().getUsers()) {
      auto subscript = dyn_cast<hc_front::SubscriptOp>(user);
      if (!subscript || subscript.getBase() != call.getResult() ||
          subscript.getIndices().size() != 1)
        return;
      std::optional<unsigned> required =
          staticLaunchGeoRequiredRank(subscript.getIndices().front());
      if (!required)
        return;
      rank = std::max(rank, *required);
    }
    if (rank != 0)
      record(call.getResult(), method, rank);
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
    return shapeRankOrZero(contextGroupShape)
               ? shapeRankOrZero(contextGroupShape)
               : groupRank.value_or(fallbackRank());
  case LaunchGeoRankDomain::Scalar:
    return fallbackRank();
  }
  llvm_unreachable("unhandled launch-geometry rank domain");
}

Value Lowerer::tryEmitLaunchGeo(const LaunchGeoMethodInfo &method,
                                Value context, Location loc,
                                std::optional<unsigned> requiredRank) {
  auto emitMulti = [&](auto tag, StringRef prefix) -> Value {
    using OpT = decltype(tag);
    unsigned n = getLaunchGeometryRank(method, context.getType(), requiredRank);
    FailureOr<SmallVector<Type>> resTypes =
        launchGeometryIdxTypes(builder.getContext(), loc, prefix, n);
    if (failed(resTypes))
      return {};
    auto op = OpT::create(builder, loc, *resTypes, context);
    auto tupleType = TupleType::get(builder.getContext(), op.getResultTypes());
    return HCTupleOp::create(builder, loc, tupleType, op.getResults());
  };
  auto emitScalar = [&](auto tag, StringRef prefix) -> Value {
    using OpT = decltype(tag);
    FailureOr<Type> resultType =
        launchGeometryIdxType(builder.getContext(), loc, prefix, 0);
    if (failed(resultType))
      return {};
    auto op = OpT::create(builder, loc, *resultType, context);
    return op.getResult(0);
  };
  switch (method.method) {
  case LaunchGeoMethod::GroupId:
    return emitMulti(HCGroupIdOp{}, method.symbolPrefix);
  case LaunchGeoMethod::LocalId:
    return emitMulti(HCLocalIdOp{}, method.symbolPrefix);
  case LaunchGeoMethod::SubgroupId:
    return emitMulti(HCSubgroupIdOp{}, method.symbolPrefix);
  case LaunchGeoMethod::GroupShape:
    return emitMulti(HCGroupShapeOp{}, method.symbolPrefix);
  case LaunchGeoMethod::WorkOffset:
    return emitMulti(HCWorkOffsetOp{}, method.symbolPrefix);
  case LaunchGeoMethod::WorkShape:
    return emitMulti(HCWorkShapeOp{}, method.symbolPrefix);
  case LaunchGeoMethod::GroupSize:
    return emitScalar(HCGroupSizeOp{}, method.symbolPrefix);
  case LaunchGeoMethod::WaveSize:
    return emitScalar(HCWaveSizeOp{}, method.symbolPrefix);
  }
  llvm_unreachable("unhandled launch-geometry method");
}

Value Lowerer::lowerSubscript(hc_front::SubscriptOp op) {
  // DSL-method `[]` patterns fold into dedicated hc ops when the base is an
  // `hc_front.attr`:
  //   x.shape[N]      -> hc.buffer_dim
  //   group.group_id[N] / local_id[N] / ... -> hc.getitem(hc.tuple(...), N)
  // Property-style (no `()`) access to launch geometry lands here. The
  // call-style form (`wi.local_id()[N]`) is handled by the sibling branch below
  // so the default type-inference schedule never sees a launch-geo value as a
  // fake buffer_view base.
  if (auto attr =
          dyn_cast_if_present<hc_front::AttrOp>(op.getBase().getDefiningOp())) {
    // Read the method name off the attr op (`$name`) rather than
    // `ref.method`: a `.shape[N]` or `.local_id[N]` chained off a
    // subscript/call result reaches here without a `ref` dict on the
    // attr, but the spelling on the op is enough — the branches below
    // only fold on exact method strings (`"shape"` and the launch-geo
    // set). Unknown spellings still fall through to the generic
    // `hc.buffer_view` path.
    StringRef method = attr.getName();
    if (op.getIndices().size() == 1) {
      FailureOr<Value> idxValOr = lowerValueOperand(
          op.getIndices().front(), op.getOperation(), "subscript index");
      if (failed(idxValOr))
        return nullptr;
      FailureOr<Value> baseValOr = lowerValueOperand(
          attr.getBase(), op.getOperation(), "subscript base");
      if (failed(baseValOr))
        return nullptr;
      Value idxVal = *idxValOr;
      Value baseVal = *baseValOr;
      auto constOp = idxVal ? idxVal.getDefiningOp<HCConstOp>() : nullptr;
      auto ax =
          constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr{};
      if (baseVal && ax) {
        int64_t axVal = ax.getInt();
        if (method == "shape") {
          // `hc.buffer_dim` axis is a buffer rank index, not a launch grid
          // rank — no small-integer cap applies. Forward the driver value;
          // bounds checking lives on the dialect verifier.
          return HCBufferDimOp::create(
              builder, op.getLoc(), undef, baseVal,
              IntegerAttr::get(IntegerType::get(op.getContext(), 64), axVal));
        }
      }
      std::optional<LaunchGeoMethodInfo> methodInfo =
          classifyLaunchGeoMethod(method);
      if (baseVal && idxVal && methodInfo) {
        if (failed(checkLaunchGeoSubscript(op, *methodInfo, ax)))
          return nullptr;
        std::optional<unsigned> requiredRank =
            getStaticLaunchGeometryRank(attr.getBase(), method);
        if (Value v = tryEmitLaunchGeo(*methodInfo, baseVal, op.getLoc(),
                                       requiredRank))
          return HCGetItemOp::create(builder, op.getLoc(), undef, v, idxVal);
      }
    }
  }
  if (auto call =
          dyn_cast_if_present<hc_front::CallOp>(op.getBase().getDefiningOp())) {
    if (auto attr = dyn_cast_if_present<hc_front::AttrOp>(
            call.getCallee().getDefiningOp())) {
      StringRef method = attr.getName();
      std::optional<LaunchGeoMethodInfo> methodInfo =
          classifyLaunchGeoMethod(method);
      if (methodInfo && call.getArguments().empty() &&
          op.getIndices().size() == 1) {
        FailureOr<Value> idxValOr = lowerValueOperand(
            op.getIndices().front(), op.getOperation(), "subscript index");
        if (failed(idxValOr))
          return nullptr;
        FailureOr<Value> launchGeoOr = lowerValueOperand(
            call.getResult(), op.getOperation(), "launch-geo call result");
        if (failed(launchGeoOr))
          return nullptr;
        Value idxVal = *idxValOr;
        Value launchGeo = *launchGeoOr;
        auto constOp = idxVal ? idxVal.getDefiningOp<HCConstOp>() : nullptr;
        auto ax =
            constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr{};
        if (launchGeo && idxVal) {
          if (failed(checkLaunchGeoSubscript(op, *methodInfo, ax)))
            return nullptr;
          return HCGetItemOp::create(builder, op.getLoc(), undef, launchGeo,
                                     idxVal);
        }
      }
    }
  }

  FailureOr<Value> baseOr =
      lowerValueOperand(op.getBase(), op.getOperation(), "subscript base");
  if (failed(baseOr))
    return nullptr;
  Value base = *baseOr;
  if (!base) {
    op.emitOpError("subscript base did not lower");
    return nullptr;
  }
  SmallVector<Value> indices;
  for (Value idx : op.getIndices()) {
    FailureOr<SmallVector<Value>> expanded =
        expandTupleOperand(idx, op.getOperation(), "subscript index");
    if (failed(expanded)) {
      op.emitOpError("subscript index did not lower");
      return nullptr;
    }
    indices.append(expanded->begin(), expanded->end());
  }
  // `hc.buffer_view` accepts `!hc.undef`, buffers, and tensors as the root
  // value, so pre-inference subscripts and inferred tensor subscripts both
  // pass verify.
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
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();
    UndefType undef = UndefType::get(ctx);

    SmallVector<Operation *> intrinsicOps;
    SmallVector<Operation *> frontOps;
    for (Operation &op : *moduleOp.getBody()) {
      if (isa<hc_front::KernelOp, hc_front::FuncOp>(op))
        frontOps.push_back(&op);
      else if (isa<hc_front::IntrinsicOp>(op))
        intrinsicOps.push_back(&op);
    }

    LaunchMetadataAttrs defaultLaunchMetadata;
    Operation *singleKernel = nullptr;
    for (Operation *op : frontOps) {
      if (!isa<hc_front::KernelOp>(op))
        continue;
      if (singleKernel) {
        singleKernel = nullptr;
        break;
      }
      singleKernel = op;
    }
    if (singleKernel)
      defaultLaunchMetadata = launchMetadataAttrsFrom(singleKernel);

    OpBuilder builder(ctx);
    for (Operation *op : intrinsicOps) {
      Lowerer lowerer(builder, undef, defaultLaunchMetadata);
      if (failed(lowerer.lowerCallable(op))) {
        signalPassFailure();
        return;
      }
      op->erase();
    }
    for (Operation *op : frontOps) {
      Lowerer lowerer(builder, undef, defaultLaunchMetadata);
      if (failed(lowerer.lowerCallable(op))) {
        signalPassFailure();
        return;
      }
      op->erase();
    }
  }
};

} // namespace

// `createConvertHCFrontToHCPass()` is emitted by tablegen (friend of
// the impl::ConvertHCFrontToHCBase CRTP). See `Passes.td` — no `let
// constructor`, so the generated factory is the only one.
