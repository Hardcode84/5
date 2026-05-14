// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-flatten-with-layouts`. Every shaped value loses its
// `#hc.layout` slot AND collapses its shape to a single entry. Tensors
// and vectors get a concrete `storage_size_expr` (from the layout's
// `storage_size` after binding `shape_syms` to the original shape, or
// the dim product when the implicit identity-layout contract
// applies); buffers collapse to `[?]` (`#hc.dyn`) because the host
// owns the allocation.
//
// In addition to the shape collapse, the converter is **1-to-N**:
// every shaped operand expands to (flat_value, idx_value_0, ...,
// idx_value_{k-1}) where the trailing values are typed `!hc.idx<sym>`
// for each free symbol the type implicitly carries — shape syms from
// the operand's dim entries, layout's free syms from the offset /
// storage_size / params expressions, minus the layout's index_syms
// which are bound at access sites instead. Names are sorted
// lexicographically to keep the expansion deterministic across
// rebuilds.
//
// The expansion is what makes the flattened IR self-contained. Pre-
// flatten the dim names (`M`, `N`, ...) and stride params
// (`$STRIDE_*`) are reachable through the type and resolved by the
// downstream launch-walk via ambient binding; post-flatten those
// names disappear from the type, so we surface them as ordinary SSA
// values that ride alongside the flat carrier. Access ops that
// compose layouts can then bind every free symbol of the resulting
// offset expression in `hc.idx_apply` explicitly, instead of leaving
// the binding to the lowering pass.
//
// `hc.generic` per-operand per-axis `#hc.expr` offset arrays compose
// post-flatten: each operand's array of axis exprs goes through the
// operand's pre-flatten layout offset (or the identity-layout
// fallback when no layout is attached) into a single 1D `#hc.expr`,
// matching the post-flatten 1D operand rank. The verifier on
// `hc.generic` enforces rank parity in both regimes.
//
// Per-access ops (`hc.load`, `hc.vload`, `hc.store`) also get
// rewritten in this pass: their multi-index lists collapse to a
// single 1D base-offset SSA value composed from the operand's
// layout against the access site's index expressions. `hc.load_mask`
// is rewritten to an `hc.generic` upstream by
// `hc-load-store-to-generic`, so its addressing rides the same
// generic-offset compose path the data generics do. Composition
// runs during conversion so the layout is still on the (pre-
// conversion) operand type when we read it. Free symbols of the
// composed offset that have a matching SSA in the operand's
// expansion (or in the index ops themselves, since `!hc.idx<sym>`
// values *are* their own binding) end up as explicit operands of
// `hc.idx_apply`; anything else stays unlisted and gets resolved
// ambiently by the launch-body lowering.
//
// `hc.buffer_view` and `hc.vec` stay untouched here — buffer_view is
// a sub-view producer with different semantics than a single base
// offset, and vec has no indices.
//
// `hc.as_layout` is dropped unconditionally — both endpoints route
// through the converter, so the relabel becomes cosmetic by the time
// the rewriter sees it.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCFLATTENWITHLAYOUTS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Collect the free symbol names a shaped type *implicitly* carries:
// names that appear in any dim expression or in the layout's
// `offset` / `storage_size` / `params` payloads, minus the layout's
// own `index_syms` (those bind at access sites, not on the type) and
// the layout's `shape_syms` (those alias to dim entries, which we
// already walked through the shape).
//
// The list is the post-flatten 1-to-N expansion's tail: for every
// name returned here the converter produces an `!hc.idx<name>`
// trailing the flat carrier, in the same order, so the access-site
// rewriters can pair operand position to symbol name without a side
// table. Sort lexicographically for determinism — `StringSet`
// iteration is unordered and the test corpus pins the operand list
// in textual IR.
static SmallVector<std::string>
collectImplicitSyms(SymbolicallyShapedTypeInterface shaped) {
  llvm::StringSet<> seen;
  auto noteExprNames = [&](ExprAttr expr) {
    if (!expr)
      return;
    sym::walkSymbolNames(expr.getValue(),
                         [&](StringRef name) { seen.insert(name); });
  };

  if (ShapeAttr shape = shaped.getSymbolicShape()) {
    for (Attribute dim : shape.getDims())
      if (auto expr = dyn_cast<ExprAttr>(dim))
        noteExprNames(expr);
  }

  if (LayoutAttr layout = shaped.getSymbolicLayout()) {
    noteExprNames(layout.getOffset());
    noteExprNames(layout.getStorageSize());
    if (DictionaryAttr params = layout.getParams())
      for (NamedAttribute entry : params.getValue())
        if (auto expr = dyn_cast<ExprAttr>(entry.getValue()))
          noteExprNames(expr);

    // index_syms bind at access sites, not on the type.
    for (Attribute attr : layout.getIndexSyms())
      seen.erase(cast<StringAttr>(attr).getValue());
    // shape_syms are layout-internal aliases for dim entries; the dim
    // walk already noted the dim names and we don't want both
    // (e.g. `d0` *and* `M`) in the expansion.
    for (Attribute attr : layout.getShapeSyms())
      seen.erase(cast<StringAttr>(attr).getValue());
  }

  SmallVector<std::string> result;
  result.reserve(seen.size());
  for (const auto &entry : seen)
    result.emplace_back(entry.getKey().str());
  llvm::sort(result);
  return result;
}

// Pull the bare symbol name out of a shape dim entry, or return an
// empty `StringRef` when the dim isn't a single-symbol expression
// (constant, composite expression, dyn-size sentinel). Used to map
// `axis -> sym name` for the host-wrapper aux-arg metadata so the
// dim-slot path can recover axis order post-flatten.
static StringRef bareDimSymbolName(Attribute dimAttr) {
  auto expr = dyn_cast<ExprAttr>(dimAttr);
  if (!expr)
    return {};
  ixs_node *node = const_cast<ixs_node *>(expr.getNode());
  if (ixs_node_tag(node) != IXS_SYM)
    return {};
  return StringRef(ixs_node_sym_name(node));
}

// `$STRIDE_<axis>_<bufname>` is the frontend's stride symbol shape (see
// `buildDefaultStridedBufferLayout`). Parsing the axis back out of the
// symbol name keeps `5-1qy4`'s host-wrapper lowering name-driven on the
// stride side, mirroring the dim side's axis lookup against the buffer
// shape. Returns `std::nullopt` for symbols that aren't strides.
static std::optional<unsigned> parseStrideAxis(StringRef name) {
  static constexpr StringLiteral kStridePrefix = "$STRIDE_";
  if (!name.starts_with(kStridePrefix))
    return std::nullopt;
  StringRef rest = name.drop_front(kStridePrefix.size());
  auto [axisStr, bufName] = rest.split('_');
  unsigned axis = 0;
  if (axisStr.consumeInteger(10, axis) || !axisStr.empty() || bufName.empty())
    return std::nullopt;
  return axis;
}

// Build the !hc.idx<sym> type carrying a bare-symbol expression
// for each implicit name. Returns failure if any name fails to
// compose into the dialect store (extreme edge case; the names
// already came out of valid type payloads).
static LogicalResult buildAuxIdxTypes(MLIRContext *ctx,
                                      ArrayRef<std::string> names,
                                      SmallVectorImpl<Type> &results) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  for (StringRef name : names) {
    auto handle = sym::composeExprSym(store, name);
    if (failed(handle))
      return failure();
    results.push_back(IdxType::get(ctx, ExprAttr::get(ctx, *handle)));
  }
  return success();
}

// Compute the 1D `storage_size_expr` for `originalShape` given an
// optional `layout`. With a layout, that is `layout.storage_size`
// after substituting `layout.shape_syms` with the original shape's
// per-axis expressions. Without a layout, the type sits on the
// implicit identity-layout contract and the storage size is the
// product of the original dimensions. Rank-0 falls out as the empty
// product `1`.
//
// Caller's responsibility: the shape's entries must all be
// `ExprAttr` (no `DynSizeAttr`). Buffers — the only flatten input
// that legitimately carries a `?` post-collapse — handle that case
// directly in the converter instead of routing through here, so
// any `DynSize` here would mean a bug upstream and the cast asserts.
//
// All work goes through hash-consed ixsimpl handles via the
// dialect-owned store: no rendering, no parsing, no string traffic.
// `ixs_subs_multi` wants raw `ixs_node *` arrays for both targets
// and replacements, so we collect those via `composeExprSym` /
// `getNode()` and let the substitution canonicalize through the same
// store any other producer of these expressions would. The actual
// composition lives in `lib/IR/HCAttrs.cpp::computeStorageSizeExpr`
// — promoted to a public helper so the `hc.as_layout` verifier
// uses the same path; the comment above documents the contract for
// both call sites.

// Pull the symbolic expression that names the SSA index value at an
// access site. Three legal sources:
//   - `!hc.idx<expr>`               -> `expr` (the typical case;
//     index ops produced by the kernel scope are pinned),
//   - `!hc.slice<lower=!hc.idx<expr>, ...>` -> the lower-bound expr
//     (slice's first element address is the access base; the tile
//     walk steps from there),
//   - `!hc.slice` with no `lower`   -> integer `0` (Python-style
//     full-slice; base offset on this axis is 0).
// Anything else (raw `index`, untyped `!hc.idx`, slice without a
// pinned lower) -> failure: the rewrite can't compose a base offset
// without a symbolic name to bind to the layout's index_sym, and
// silently leaving the multi-index list alone would mismatch the
// op's now-1D operand type.
static FailureOr<ExprAttr>
extractAccessIndexExpr(MLIRContext *ctx, sym::Store &store, Type indexType) {
  if (auto idx = llvm::dyn_cast<IdxType>(indexType)) {
    if (ExprAttr expr = idx.getExpr())
      return expr;
    return failure();
  }
  if (auto slice = llvm::dyn_cast<SliceType>(indexType)) {
    Type lowerTy = slice.getLowerType();
    if (!lowerTy) {
      auto zero = sym::composeExprInt(store, 0);
      if (failed(zero))
        return failure();
      return ExprAttr::get(ctx, *zero);
    }
    if (auto lowerIdx = llvm::dyn_cast<IdxType>(lowerTy))
      if (ExprAttr expr = lowerIdx.getExpr())
        return expr;
    return failure();
  }
  return failure();
}

// Identity-layout offset and `composeAccessOffsetExpr` itself live in
// `lib/IR/HCAttrs.cpp` as `mlir::hc::composeAccessOffsetExpr` so other
// passes (notably `hc-load-store-to-generic`'s non-injective vload
// path) reuse the same substitution and stay aligned with this pass
// on hash-consed offset handles.

// Materialize the composed offset as an SSA value typed
// `!hc.idx<offset_expr>` via an `hc.idx_apply`. Free symbols of the
// expression that have a known SSA binding in `bindings` are listed
// explicitly (operand + symbol name pair); anything else stays
// unlisted and gets resolved ambiently by the launch-body lowering
// from the surrounding launch context. Listed symbol order is
// lexicographic so the textual form is deterministic across
// rebuilds.
static Value materializeOffsetSSA(ConversionPatternRewriter &rewriter,
                                  Location loc, ExprAttr offsetExpr,
                                  const llvm::StringMap<Value> &bindings) {
  llvm::StringSet<> freeSyms;
  sym::walkSymbolNames(offsetExpr.getValue(),
                       [&](StringRef name) { freeSyms.insert(name); });

  SmallVector<StringRef> listed;
  for (const auto &entry : bindings)
    if (freeSyms.contains(entry.getKey()))
      listed.push_back(entry.getKey());
  llvm::sort(listed);

  SmallVector<Attribute> symAttrs;
  SmallVector<Value> operands;
  symAttrs.reserve(listed.size());
  operands.reserve(listed.size());
  for (StringRef name : listed) {
    symAttrs.push_back(rewriter.getStringAttr(name));
    operands.push_back(bindings.lookup(name));
  }

  auto idxType = IdxType::get(rewriter.getContext(), offsetExpr);
  return HCIdxApplyOp::create(rewriter, loc, idxType, operands,
                              rewriter.getArrayAttr(symAttrs))
      .getResult();
}

// Per-access-op helper: extracts each index operand's symbolic expr,
// composes the base offset against the operand's pre-flatten layout
// + shape, materializes the offset as a single SSA value with
// explicit bindings for every free symbol whose SSA we have at
// hand. The buffer/source operand's 1-to-N expansion (`shapedAux`,
// in declaration order matching `collectImplicitSyms` of the
// operand's pre-flatten type) supplies the dim and stride values;
// each idx-typed index operand binds its own symbol because the
// `!hc.idx<sym>` value *is* the binding for that name. Returns
// failure (and leaves the IR untouched) if any index can't yield a
// symbolic expression, or if the operand isn't a shaped HC type, or
// if rank parity breaks. Failure here means the access op stays on
// the multi-index surface; `RetypeAnyHCOp` then handles type-only
// retyping and the lowering downstream still has to deal with the
// uncomposed access.
static FailureOr<Value>
composeAccessBaseOffset(ConversionPatternRewriter &rewriter, Operation *op,
                        Value preFlattenOperand, ValueRange shapedAux,
                        OperandRange indices) {
  auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(
      preFlattenOperand.getType());
  if (!shaped)
    return failure();
  ShapeAttr originalShape = shaped.getSymbolicShape();
  if (!originalShape)
    return failure();
  // Buffers ride on a `[?]` post-flatten shape and an absent layout
  // here would mean we'd fall back to the identity layout over the
  // wrong dims. Today every buffer carries the default strided
  // layout, so a missing layout on a buffer is a frontend bug we
  // surface as a rewrite failure instead of silently emitting `0`.
  LayoutAttr layout = shaped.getSymbolicLayout();
  if (!layout && llvm::isa<BufferType>(preFlattenOperand.getType()))
    return failure();
  for (Attribute dim : originalShape.getDims())
    if (!llvm::isa<ExprAttr>(dim))
      return failure();

  MLIRContext *ctx = op->getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallVector<ExprAttr> indexExprs;
  indexExprs.reserve(indices.size());
  for (Value idx : indices) {
    auto expr = extractAccessIndexExpr(ctx, store, idx.getType());
    if (failed(expr))
      return failure();
    indexExprs.push_back(*expr);
  }
  auto offsetExpr =
      composeAccessOffsetExpr(ctx, layout, originalShape, indexExprs);
  if (failed(offsetExpr))
    return failure();

  // Build the symbol-name → SSA map from (a) the operand's expansion
  // — its trailing aux values are typed `!hc.idx<sym>` for each name
  // in `collectImplicitSyms` order — and (b) every idx-typed index
  // operand whose own type pins a single bare symbol that the
  // composed offset references.
  llvm::StringMap<Value> bindings;
  SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
  // Defensive: the converter is supposed to produce the operand and
  // exactly one aux per implicit sym. A mismatch means someone fed us
  // a partially-converted operand range; bail rather than guess.
  if (shapedAux.size() != implicitSyms.size())
    return failure();
  for (auto [name, value] : llvm::zip_equal(implicitSyms, shapedAux))
    bindings[name] = value;

  // Only bind when the type pins a *bare* free symbol. Composite
  // expressions (`i + 1`, `i * stride`) are not their own binding for
  // any single name; the value-as-binding shortcut only applies when
  // the type's symbol set is exactly `{name}` and the expression *is*
  // that symbol leaf. The cheapest check: walk and single-out a
  // unique name, then confirm by reconstruction.
  auto pinsBareSymbol = [&store](Type type) -> StringRef {
    auto idxType = dyn_cast<IdxType>(type);
    if (!idxType)
      return {};
    ExprAttr expr = idxType.getExpr();
    if (!expr)
      return {};
    StringRef onlyName;
    bool unique = true;
    sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
      if (onlyName.empty())
        onlyName = name;
      else if (onlyName != name)
        unique = false;
    });
    if (!unique || onlyName.empty())
      return {};
    auto pinned = sym::composeExprSym(store, onlyName);
    if (failed(pinned))
      return {};
    if (pinned->raw() != expr.getValue().raw())
      return {};
    return onlyName;
  };

  for (Value idx : indices) {
    StringRef name = pinsBareSymbol(idx.getType());
    if (name.empty())
      continue;
    // Don't overwrite a binding from the operand's expansion.
    bindings.try_emplace(name, idx);
  }

  // Walk enclosing region/loop block arguments (the canonical example
  // is an `hc.for_range` induction variable typed
  // `!hc.idx<"$join0">`) and bind any bare-sym `!hc.idx` we find.
  // Without this the composed offset would leave such names as free
  // symbols, and the launch-body lowering would have to resolve them
  // ambiently — fragile, because the structured-loop converter
  // rewrites the for_range to `scf.for` before the inner apply gets
  // lowered, and at that point the original `!hc.idx<sym>` type is
  // gone from the IR. Explicit operand binding here keeps the apply's
  // free-sym set bounded to the launch geometry and kernel-arg shape
  // syms.
  for (Block *block = op->getBlock(); block;) {
    for (BlockArgument arg : block->getArguments()) {
      StringRef name = pinsBareSymbol(arg.getType());
      if (name.empty())
        continue;
      bindings.try_emplace(name, arg);
    }
    Operation *parent = block->getParentOp();
    if (!parent)
      break;
    block = parent->getBlock();
  }

  return materializeOffsetSSA(rewriter, op->getLoc(), *offsetExpr, bindings);
}

// Decides whether a shaped type is already in its post-flatten
// canonical form: no layout slot, and a single-entry shape (an
// `ExprAttr` for tensors / vectors, the `?` sentinel for buffers).
// Already-flat types convert 1-to-1 — re-running the expansion on
// them would never reach a fixed point because the converter would
// keep producing a non-trivial 1-to-N expansion for the same type.
static bool isAlreadyFlat(SymbolicallyShapedTypeInterface shaped) {
  if (shaped.getSymbolicLayout())
    return false;
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return true;
  ArrayRef<Attribute> dims = shape.getDims();
  if (dims.size() != 1)
    return false;
  if (isa<BufferType>(Type(shaped)))
    return isa<DynSizeAttr>(dims.front());
  return isa<ExprAttr>(dims.front());
}

// Compute the 1D shaped form of a `SymbolicallyShapedTypeInterface`
// (the leading flat carrier the converter prepends to the implicit
// sym aux). Returns std::nullopt for types the conversion declines
// (e.g. an unresolvable `storage_size`).
static std::optional<Type>
computeFlatShapedType(SymbolicallyShapedTypeInterface shaped) {
  ShapeAttr originalShape = shaped.getSymbolicShape();
  if (!originalShape)
    return std::nullopt;

  MLIRContext *ctx = shaped.getContext();
  LayoutAttr layout = shaped.getSymbolicLayout();

  // Buffers collapse to `[?]`: the host owns the allocation and
  // the default strided layout's `storage_size = 0` placeholder
  // is informational, so a `#hc.dyn` sentinel is the honest
  // 1D form. Any consumer that needs a concrete extent reaches
  // for the host descriptor instead of trying to derive it from
  // the in-IR symbol set.
  if (isa<BufferType>(Type(shaped))) {
    ShapeAttr collapsed = ShapeAttr::get(ctx, {DynSizeAttr::get(ctx)});
    if (collapsed == originalShape && !layout)
      return Type(shaped);
    Type withShape = shaped.cloneWithSymbolicShape(collapsed);
    auto reshaped = cast<SymbolicallyShapedTypeInterface>(withShape);
    return reshaped.cloneWithSymbolicLayout(LayoutAttr{});
  }

  if (isAlreadyFlat(shaped))
    return Type(shaped);

  FailureOr<ExprAttr> storageSize =
      computeStorageSizeExpr(ctx, layout, originalShape);
  if (failed(storageSize))
    return std::nullopt;
  ShapeAttr collapsed = ShapeAttr::get(ctx, {*storageSize});
  Type withShape = shaped.cloneWithSymbolicShape(collapsed);
  auto reshaped = cast<SymbolicallyShapedTypeInterface>(withShape);
  return reshaped.cloneWithSymbolicLayout(LayoutAttr{});
}

// 1-to-N `TypeConverter` for the flatten pass. Every
// `SymbolicallyShapedTypeInterface` value expands to (flat shaped
// type, aux idx for each implicit free symbol). The 1-to-N model
// makes every dynamic dim and stride sym visible as ordinary SSA
// post-flatten so access ops can bind them explicitly in
// `hc.idx_apply`, instead of leaking the type's name list into the
// downstream launch-walk.
//
// For shape-preserving non-access HC ops the catch-all retyper feeds
// only the leading flat value into the new op and threads the
// trailing aux through to the result expansion (the result and the
// operand share the same implicit sym set because shape is
// preserved). Shape-changing rewriters whose result aux can't be
// sourced from any operand fall back to ambient (unlisted)
// `hc.idx_apply` per missing sym, mirroring today's
// materialize-bound-exprs behavior on those names.
//
// Tuples and function types recurse via `convertTypes`; both can
// host shaped element/argument types and propagate the expansion
// transparently. The catch-all identity is registered first so the
// shaped / tuple / function overrides take precedence on dispatch
// (last-registered-wins).
//
// `unrealized_conversion_cast` materializations bridge the ABI: the
// 1-to-N target form takes the original value and produces N
// converted values; the source form takes N values back to one.
// Surviving casts on a boundary nothing converted fold via
// `reconcile-unrealized-casts` downstream.
class FlattenLayoutConverter : public TypeConverter {
public:
  FlattenLayoutConverter() {
    addConversion([](Type t, SmallVectorImpl<Type> &results) {
      results.push_back(t);
      return success();
    });

    addConversion([](SymbolicallyShapedTypeInterface shaped,
                     SmallVectorImpl<Type> &results)
                      -> std::optional<LogicalResult> {
      // Already-flat types convert 1-to-1: no layout, single-entry
      // shape, and any free symbols inside that single dim
      // expression are now part of the storage-size expression and
      // not re-exposed as separate aux. Re-running the expansion on
      // them would never reach a fixed point because the converter
      // would keep emitting the same 1-to-N tuple for the same type.
      if (isAlreadyFlat(shaped)) {
        results.push_back(Type(shaped));
        return success();
      }

      std::optional<Type> flat = computeFlatShapedType(shaped);
      if (!flat)
        return std::nullopt;
      results.push_back(*flat);

      SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
      if (failed(buildAuxIdxTypes(shaped.getContext(), implicitSyms, results)))
        return failure();
      return success();
    });

    addConversion(
        [this](TupleType tuple,
               SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          SmallVector<Type> elements;
          if (failed(convertTypes(tuple.getTypes(), elements)))
            return failure();
          results.push_back(TupleType::get(tuple.getContext(), elements));
          return success();
        });

    addConversion(
        [this](FunctionType fn,
               SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          SmallVector<Type> ins;
          SmallVector<Type> outs;
          if (failed(convertTypes(fn.getInputs(), ins)))
            return failure();
          if (failed(convertTypes(fn.getResults(), outs)))
            return failure();
          results.push_back(FunctionType::get(fn.getContext(), ins, outs));
          return success();
        });

    auto sourceMat = [](OpBuilder &builder, Type resultType, ValueRange inputs,
                        Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    auto targetMat = [](OpBuilder &builder, TypeRange resultTypes,
                        ValueRange inputs, Location loc) -> SmallVector<Value> {
      auto cast =
          UnrealizedConversionCastOp::create(builder, loc, resultTypes, inputs);
      return SmallVector<Value>(cast.getResults());
    };
    addSourceMaterialization(sourceMat);
    addTargetMaterialization(targetMat);
  }
};

// `hc.as_layout` was the layout-relabel surface op. Once both endpoints
// route through `FlattenLayoutConverter`, the op is purely cosmetic —
// the source value already has the converted type that the result is
// asking for. Drop the op and replace its result with the converted
// operand expansion (the 1-to-N adapter forwards every value the
// converter emitted, including any aux idx values).
struct DropAsLayout : public OpConversionPattern<HCAsLayoutOp> {
  using OpConversionPattern::OpConversionPattern;
  using Base = OpConversionPattern<HCAsLayoutOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCAsLayoutOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<ValueRange> replacement = {adaptor.getValue()};
    rewriter.replaceOpWithMultiple(op, replacement);
    return success();
  }
};

// Re-key a per-result aux-value lookup table from a converted operand
// expansion. The input `operandRange` is the 1-to-N adapter's value
// range for one original operand; the leading entry is the flat
// shaped value, the trailing entries match the operand's pre-flatten
// type's `collectImplicitSyms` order. We thread the trailing values
// into `bindings` keyed by symbol name so a downstream lookup can
// pull the SSA for a given name without re-running the converter.
// Operands with empty implicit-sym lists (constant-shape vectors,
// scalars, `!hc.idx<sym>` carriers) contribute nothing.
static void noteOperandBindings(Type preFlattenType, ValueRange operandRange,
                                llvm::StringMap<Value> &bindings) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(preFlattenType);
  if (!shaped)
    return;
  SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
  if (implicitSyms.empty())
    return;
  if (operandRange.size() != implicitSyms.size() + 1)
    return;
  for (auto [name, value] :
       llvm::zip_equal(implicitSyms, operandRange.drop_front()))
    bindings.try_emplace(name, value);
}

// Identify a type that pins a single bare symbol — used to recognize
// `!hc.idx<"$WG0">`-style block args / SSAs whose own type IS the
// binding for one name. Composite expressions (`i + 1`, `i * stride`)
// are not their own binding; the value-as-binding shortcut only
// applies when the type's symbol set is exactly `{name}` and the
// expression is that symbol leaf. Returns empty `StringRef` if the
// type doesn't qualify.
static StringRef typePinsBareSymbol(sym::Store &store, Type type) {
  auto idxType = dyn_cast<IdxType>(type);
  if (!idxType)
    return {};
  ExprAttr expr = idxType.getExpr();
  if (!expr)
    return {};
  StringRef onlyName;
  bool unique = true;
  sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
    if (onlyName.empty())
      onlyName = name;
    else if (onlyName != name)
      unique = false;
  });
  if (!unique || onlyName.empty())
    return {};
  auto pinned = sym::composeExprSym(store, onlyName);
  if (failed(pinned))
    return {};
  if (pinned->raw() != expr.getValue().raw())
    return {};
  return onlyName;
}

// Walk enclosing region/loop block arguments and operand-result SSAs
// produced by ancestor ops, binding every `!hc.idx<"$name">` value we
// find to its bare symbol name. Canonical sources: `hc.for_range`'s
// induction var (typed `!hc.idx<"$joinN">`), `gpu.launch`'s block
// args ($WG*, $WI*, $WGS*), buffer-from-ptr UCC retype aux outputs
// ($STRIDE_*, dim syms), and other `hc.idx_apply` results that
// happen to land in scope.
//
// `bindings.try_emplace` preserves the first binding wins rule —
// callers that pre-seed the map (with op-local 1-to-N operand
// expansions) keep precedence over ancestor scans.
static void collectAncestorIdxBindings(sym::Store &store, Operation *op,
                                       llvm::StringMap<Value> &bindings) {
  for (Block *block = op->getBlock(); block;) {
    for (BlockArgument arg : block->getArguments()) {
      StringRef name = typePinsBareSymbol(store, arg.getType());
      if (!name.empty())
        bindings.try_emplace(name, arg);
    }
    // Also pick up `!hc.idx<sym>` results from ops preceding `op` in
    // its own block — the kernel-arg-bundle retype UCC plants these,
    // and they're the SSA value for `$STRIDE_*` / dim syms in scope
    // for any op that comes after it. Pre-`op` SSAs in ancestor
    // blocks were already covered by traversing parent->block->ops.
    for (Operation &prev : *block) {
      if (&prev == op)
        break;
      for (Value res : prev.getResults()) {
        StringRef name = typePinsBareSymbol(store, res.getType());
        if (!name.empty())
          bindings.try_emplace(name, res);
      }
    }
    Operation *parent = block->getParentOp();
    if (!parent)
      break;
    block = parent->getBlock();
  }
}

// Resolve the trailing aux value for each implicit sym a result type
// carries. The first matching operand expansion wins; missing
// names fall through to an empty-binding `hc.idx_apply` sourced
// against the kernel's ambient bindings (same severance form
// `materialize-bound-exprs` lays down). Returns one `Value` per
// trailing aux in the result's expansion, in `collectImplicitSyms`
// order — the caller pairs these with the new op's flat result to
// form the `replaceOpWithMultiple` argument.
static FailureOr<SmallVector<Value>>
resolveResultAuxValues(ConversionPatternRewriter &rewriter, Location loc,
                       Type preFlattenResult,
                       const llvm::StringMap<Value> &operandBindings) {
  SmallVector<Value> auxValues;
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(preFlattenResult);
  if (!shaped)
    return auxValues;
  SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
  if (implicitSyms.empty())
    return auxValues;

  MLIRContext *ctx = rewriter.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  auxValues.reserve(implicitSyms.size());
  for (StringRef name : implicitSyms) {
    auto it = operandBindings.find(name);
    if (it != operandBindings.end()) {
      auxValues.push_back(it->second);
      continue;
    }
    // No operand carries this name: emit an unlisted `hc.idx_apply`
    // pinned to the bare symbol. Lower-launch-body resolves it
    // against the kernel's ambient bindings the same way the old
    // empty-binding severance did.
    auto handle = sym::composeExprSym(store, name);
    if (failed(handle))
      return failure();
    auto idxTy = IdxType::get(ctx, ExprAttr::get(ctx, *handle));
    auxValues.push_back(HCIdxApplyOp::create(rewriter, loc, idxTy, ValueRange{},
                                             rewriter.getArrayAttr({}))
                            .getResult());
  }
  return auxValues;
}

// Slice the leading flat values out of a 1-to-N operand range so a
// rebuilt op only sees the carrier types. The trailing aux values
// flow through to result expansions via the binding map and don't
// belong on the new op's operand list.
static SmallVector<Value> flatOperandsOnly(ArrayRef<ValueRange> operands) {
  SmallVector<Value> flatOnly;
  flatOnly.reserve(operands.size());
  for (ValueRange range : operands) {
    if (range.empty())
      continue;
    flatOnly.push_back(range.front());
  }
  return flatOnly;
}

// Compose multi-index access ops down to a single 1D base offset. Pre-
// flatten the operand's type still carries the layout (and the original
// nD shape); we read both off `op.get<Operand>().getType()`, build the
// composed offset (binding every free symbol whose SSA we can source
// from the buffer/source operand's expansion or from the index ops
// themselves), and rebuild the op with the converted (1D) operand
// type and a one-element index list.
//
// Each pattern bumps benefit above the generic `RetypeAnyHCOp` so the
// driver picks it first; on failure (unbound index, untyped operand,
// rank mismatch) the rewrite signals failure and `RetypeAnyHCOp` does
// the type-only retyping fallback — the access op stays on the nD
// index list and the consuming pass owns the uncomposed access.
//
// Per-op pattern bodies are nearly identical except for the operand
// shape (which fields carry the buffer / source / dest, the indices,
// and the optional shape / source / mask passthroughs); a small CRTP
// base would shrink the boilerplate but obscures which ODS attribute
// each op carries. Four short rewrites are easier to read.

template <typename OpT>
struct ComposeAccessOffsetBase : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  ComposeAccessOffsetBase(const TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<OpT>(converter, ctx, /*benefit=*/2) {}
};

// Drain a 1-to-N adapter's variadic operand range to a flat
// `Value` list, asserting each sub-range is single-valued. Used
// for index lists where every entry is an `index` or `!hc.idx<...>`
// scalar that the converter passes through unchanged.
static FailureOr<SmallVector<Value>>
collectScalarOperands(ArrayRef<ValueRange> operands) {
  SmallVector<Value> values;
  values.reserve(operands.size());
  for (ValueRange range : operands) {
    if (range.size() != 1)
      return failure();
    values.push_back(range.front());
  }
  return values;
}

struct ComposeLoadOffsets : public ComposeAccessOffsetBase<HCLoadOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCLoadOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Already-flat single-index loads against a layout-less buffer
    // need no offset folding — the generic retype handles the operand
    // type. Layout-bearing 1D sources (the buffer_view-with-strided-
    // slice residual that `composeBufferViewLayout` produces) still
    // need composition, so keep going when the pre-flatten source's
    // type carries a layout.
    if (op.getIndices().size() == 1) {
      auto shaped =
          dyn_cast<SymbolicallyShapedTypeInterface>(op.getBuffer().getType());
      if (!shaped || !shaped.getSymbolicLayout())
        return failure();
    }
    if (adaptor.getBuffer().empty())
      return failure();
    Value flatBuffer = adaptor.getBuffer().front();
    ValueRange bufferAux = adaptor.getBuffer().drop_front();

    auto indices = collectScalarOperands(adaptor.getIndices());
    if (failed(indices))
      return failure();
    if (adaptor.getShape().size() != 1)
      return failure();

    auto base = composeAccessBaseOffset(rewriter, op, op.getBuffer(), bufferAux,
                                        op.getIndices());
    if (failed(base))
      return failure();

    Type origResultType = op.getResult().getType();
    SmallVector<Type> convertedResults;
    if (failed(
            getTypeConverter()->convertType(origResultType, convertedResults)))
      return failure();
    if (convertedResults.empty())
      return failure();

    auto newLoad =
        HCLoadOp::create(rewriter, op.getLoc(), convertedResults.front(),
                         flatBuffer, ValueRange{*base}, adaptor.getShape()[0],
                         /*layout=*/LayoutAttr{});

    llvm::StringMap<Value> bindings;
    noteOperandBindings(op.getBuffer().getType(), adaptor.getBuffer(),
                        bindings);
    auto auxValues =
        resolveResultAuxValues(rewriter, op.getLoc(), origResultType, bindings);
    if (failed(auxValues))
      return failure();

    SmallVector<Value> replacement = {newLoad.getResult()};
    llvm::append_range(replacement, *auxValues);
    SmallVector<ValueRange> replacements = {replacement};
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

struct ComposeVLoadOffsets : public ComposeAccessOffsetBase<HCVLoadOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCVLoadOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCVLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // See `ComposeLoadOffsets` — 1D layout-bearing sources (the
    // buffer_view-with-strided-slice residual that
    // `composeBufferViewLayout` produces) still need composition.
    if (op.getIndices().size() == 1) {
      auto shaped =
          dyn_cast<SymbolicallyShapedTypeInterface>(op.getSource().getType());
      if (!shaped || !shaped.getSymbolicLayout())
        return failure();
    }
    if (adaptor.getSource().empty())
      return failure();
    Value flatSource = adaptor.getSource().front();
    ValueRange sourceAux = adaptor.getSource().drop_front();

    auto indices = collectScalarOperands(adaptor.getIndices());
    if (failed(indices))
      return failure();
    if (adaptor.getShape().size() != 1)
      return failure();

    auto base = composeAccessBaseOffset(rewriter, op, op.getSource(), sourceAux,
                                        op.getIndices());
    if (failed(base))
      return failure();

    Type origResultType = op.getResult().getType();
    SmallVector<Type> convertedResults;
    if (failed(
            getTypeConverter()->convertType(origResultType, convertedResults)))
      return failure();
    if (convertedResults.empty())
      return failure();

    auto newVLoad =
        HCVLoadOp::create(rewriter, op.getLoc(), convertedResults.front(),
                          flatSource, ValueRange{*base}, adaptor.getShape()[0],
                          /*layout=*/LayoutAttr{});

    llvm::StringMap<Value> bindings;
    noteOperandBindings(op.getSource().getType(), adaptor.getSource(),
                        bindings);
    auto auxValues =
        resolveResultAuxValues(rewriter, op.getLoc(), origResultType, bindings);
    if (failed(auxValues))
      return failure();

    SmallVector<Value> replacement = {newVLoad.getResult()};
    llvm::append_range(replacement, *auxValues);
    SmallVector<ValueRange> replacements = {replacement};
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

struct ComposeStoreOffsets : public ComposeAccessOffsetBase<HCStoreOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCStoreOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // See `ComposeLoadOffsets` — 1D layout-bearing destinations still
    // need composition.
    if (op.getIndices().size() == 1) {
      auto shaped =
          dyn_cast<SymbolicallyShapedTypeInterface>(op.getDest().getType());
      if (!shaped || !shaped.getSymbolicLayout())
        return failure();
    }
    if (adaptor.getDest().empty())
      return failure();
    Value flatDest = adaptor.getDest().front();
    ValueRange destAux = adaptor.getDest().drop_front();

    auto indices = collectScalarOperands(adaptor.getIndices());
    if (failed(indices))
      return failure();
    if (adaptor.getSource().empty())
      return failure();
    Value flatSource = adaptor.getSource().front();
    Value mask;
    if (!adaptor.getMask().empty()) {
      if (adaptor.getMask().size() < 1)
        return failure();
      mask = adaptor.getMask().front();
    }

    auto base = composeAccessBaseOffset(rewriter, op, op.getDest(), destAux,
                                        op.getIndices());
    if (failed(base))
      return failure();

    HCStoreOp::create(rewriter, op.getLoc(), flatDest, ValueRange{*base},
                      flatSource, mask);
    rewriter.eraseOp(op);
    return success();
  }
};

// Rewrite an `hc.buffer_view` whose source has been flattened to its 1D
// carrier. The pre-flatten subscript list is rank-N (one entry per
// source axis); post-flatten the source carries a single composite
// axis so the verifier on the consumer side rejects the original
// subscript count.
//
// Two cases:
//
// * Identity: the converted (1D) source shape matches the converted
//   (1D) result shape. The buffer_view is a no-op against the
//   collapsed storage — every scalar subscript landed on an axis that
//   the flatten layout already factors out (unit-size axis, or a
//   work-distributed axis whose lane index is implicit per thread).
//   Forward the flat source through.
//
// * Single-slice strided: exactly one subscript is a slice; the rest
//   are scalar `!hc.idx<...>` indices. Compose the identity-layout
//   offset into a single contiguous-stride slice on the flat carrier.
//   Layout-less sources are the canonical contract for this rewrite;
//   sources carrying an explicit `#hc.layout<...>` payload bail (the
//   v0 surface uses the identity layout for these views — a custom
//   layout would need to participate in the offset composition the
//   same way `composeAccessOffsetExpr` does for access ops, which is
//   out of scope here).
//
// Anything else (multi-slice on a non-trivial layout, missing
// `hc.slice_expr` producer for the slice operand, ...) bails to the
// catch-all retyper. The remaining producer is `hc-vec`-style WMMA
// tile loads through workgroup-staged LDS.
struct ComposeBufferViewOffsets
    : public ComposeAccessOffsetBase<HCBufferViewOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCBufferViewOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCBufferViewOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getBuffer().empty())
      return failure();
    Value flatSource = adaptor.getBuffer().front();
    ValueRange sourceAux = adaptor.getBuffer().drop_front();

    auto preFlattenSrc =
        dyn_cast<SymbolicallyShapedTypeInterface>(op.getBuffer().getType());
    if (!preFlattenSrc)
      return failure();
    ShapeAttr preShape = preFlattenSrc.getSymbolicShape();
    if (!preShape)
      return failure();
    if (preShape.getDims().size() == op.getIndices().size() &&
        preShape.getDims().size() <= 1)
      return failure();
    // Non-layout sources still need the full-bind rank parity here: the
    // strided-slice branch below only knows how to fold an `[indices..]`
    // subscript stream when every axis is named. Layout-bearing sources
    // route through the identity branch instead — the residual layout
    // composed at `hc.buffer_view` type inference already bakes the
    // scalar-axis substitutions into the result's offset, so the flat
    // carriers on both sides describe the same physical storage and
    // forwarding through is correct regardless of how many axes the
    // subscript stream consumed.
    if (!preFlattenSrc.getSymbolicLayout() &&
        preShape.getDims().size() != op.getIndices().size())
      return failure();

    Type origResultType = op.getResult().getType();
    SmallVector<Type> convertedResults;
    if (failed(
            getTypeConverter()->convertType(origResultType, convertedResults)))
      return failure();
    if (convertedResults.empty())
      return failure();
    Type flatResultType = convertedResults.front();

    auto flatResultShaped =
        dyn_cast<SymbolicallyShapedTypeInterface>(flatResultType);
    auto flatSourceShaped =
        dyn_cast<SymbolicallyShapedTypeInterface>(flatSource.getType());
    if (!flatResultShaped || !flatSourceShaped)
      return failure();

    auto pushReplacement = [&](Value flatValue) {
      llvm::StringMap<Value> bindings;
      noteOperandBindings(op.getBuffer().getType(), adaptor.getBuffer(),
                          bindings);
      auto auxValues = resolveResultAuxValues(rewriter, op.getLoc(),
                                              origResultType, bindings);
      if (failed(auxValues))
        return failure();
      SmallVector<Value> replacement = {flatValue};
      llvm::append_range(replacement, *auxValues);
      SmallVector<ValueRange> replacements = {replacement};
      rewriter.replaceOpWithMultiple(op, replacements);
      return success();
    };

    // Identity: the flat carriers match between source and result, so
    // the view describes the same physical storage. Two distinct cases
    // converge here. (1) Full-bind on a layout-less source where every
    // scalar subscript hits an axis the surrounding flatten layout has
    // already factored out; the source-side strides land on the same
    // 1-D carrier the result wants. (2) Layout-bearing source where
    // `inferBufferViewResult` substituted scalar-axis index values into
    // the result's offset and dropped the corresponding shape syms —
    // residual `storage_size` still names the same physical span, the
    // flat carrier types match, and the only delta is the relabel of
    // the per-axis aux set, which `resolveResultAuxValues` rebinds from
    // the source's expansion. Cross-element-type view requests don't
    // exist in the v0 surface; reject any flat-shape coincidence that
    // changes the element type before it can silently miscompile.
    if (flatSourceShaped.getSymbolicShape() ==
            flatResultShaped.getSymbolicShape() &&
        flatSource.getType() == flatResultType) {
      (void)sourceAux;
      return pushReplacement(flatSource);
    }

    // Strided slice: exactly one slice subscript, rest scalar.
    if (preFlattenSrc.getSymbolicLayout())
      return failure();
    int64_t sliceAxis = -1;
    for (auto [axis, idx] : llvm::enumerate(op.getIndices())) {
      if (isa<SliceType>(idx.getType())) {
        if (sliceAxis >= 0)
          return failure();
        sliceAxis = static_cast<int64_t>(axis);
      } else if (!isa<IdxType>(idx.getType())) {
        return failure();
      }
    }
    if (sliceAxis < 0)
      return failure();

    MLIRContext *ctx = op.getContext();
    auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
    ArrayRef<Attribute> preDims = preShape.getDims();

    auto getDimExpr = [&](size_t axis) -> FailureOr<sym::ExprHandle> {
      auto e = dyn_cast<ExprAttr>(preDims[axis]);
      if (!e)
        return failure();
      return e.getValue();
    };

    auto rowStrideAt = [&](size_t k) -> FailureOr<sym::ExprHandle> {
      auto oneE = sym::composeExprInt(store, 1);
      if (failed(oneE))
        return failure();
      sym::ExprHandle stride = *oneE;
      for (size_t j = k + 1; j < preDims.size(); ++j) {
        auto dim = getDimExpr(j);
        if (failed(dim))
          return failure();
        auto next =
            sym::composeExprBinary(store, stride, sym::ExprBinaryOp::Mul, *dim);
        if (failed(next))
          return failure();
        stride = *next;
      }
      return stride;
    };

    auto zeroExpr = sym::composeExprInt(store, 0);
    auto oneExpr = sym::composeExprInt(store, 1);
    if (failed(zeroExpr) || failed(oneExpr))
      return failure();

    // Accumulate the scalar-axis contribution to the flat base offset.
    sym::ExprHandle baseOffset = *zeroExpr;
    for (auto [axis, idx] : llvm::enumerate(op.getIndices())) {
      if (axis == static_cast<size_t>(sliceAxis))
        continue;
      auto idxType = dyn_cast<IdxType>(idx.getType());
      if (!idxType || !idxType.getExpr())
        return failure();
      auto rowStride = rowStrideAt(axis);
      if (failed(rowStride))
        return failure();
      auto term = sym::composeExprBinary(store, idxType.getExpr().getValue(),
                                         sym::ExprBinaryOp::Mul, *rowStride);
      if (failed(term))
        return failure();
      auto added = sym::composeExprBinary(store, baseOffset,
                                          sym::ExprBinaryOp::Add, *term);
      if (failed(added))
        return failure();
      baseOffset = *added;
    }

    // Pull lower / upper / step off the slice subscript's producing
    // `hc.slice_expr`. Optional operands default to Python slice
    // semantics: lower → 0, upper → axis size, step → 1.
    auto sliceProducer =
        op.getIndices()[sliceAxis].getDefiningOp<HCSliceExprOp>();
    if (!sliceProducer)
      return failure();
    auto exprFromOperand =
        [&](Value v, sym::ExprHandle dflt) -> FailureOr<sym::ExprHandle> {
      if (!v)
        return dflt;
      auto t = dyn_cast<IdxType>(v.getType());
      if (!t || !t.getExpr())
        return failure();
      return t.getExpr().getValue();
    };
    auto sliceLower = exprFromOperand(sliceProducer.getLower(), *zeroExpr);
    if (failed(sliceLower))
      return failure();
    auto sliceAxisDim = getDimExpr(sliceAxis);
    if (failed(sliceAxisDim))
      return failure();
    auto sliceUpper = exprFromOperand(sliceProducer.getUpper(), *sliceAxisDim);
    if (failed(sliceUpper))
      return failure();
    auto sliceStep = exprFromOperand(sliceProducer.getStep(), *oneExpr);
    if (failed(sliceStep))
      return failure();

    auto sliceRowStride = rowStrideAt(sliceAxis);
    if (failed(sliceRowStride))
      return failure();

    // Flat slice lower = scalar_base + slice.lower * row_stride.
    auto sliceLowerContrib = sym::composeExprBinary(
        store, *sliceLower, sym::ExprBinaryOp::Mul, *sliceRowStride);
    if (failed(sliceLowerContrib))
      return failure();
    auto flatLowerExpr = sym::composeExprBinary(
        store, baseOffset, sym::ExprBinaryOp::Add, *sliceLowerContrib);
    if (failed(flatLowerExpr))
      return failure();

    // Flat slice upper = scalar_base + slice.upper * row_stride.
    auto sliceUpperContrib = sym::composeExprBinary(
        store, *sliceUpper, sym::ExprBinaryOp::Mul, *sliceRowStride);
    if (failed(sliceUpperContrib))
      return failure();
    auto flatUpperExpr = sym::composeExprBinary(
        store, baseOffset, sym::ExprBinaryOp::Add, *sliceUpperContrib);
    if (failed(flatUpperExpr))
      return failure();

    // Flat slice step = slice.step * row_stride.
    auto flatStepExpr = sym::composeExprBinary(
        store, *sliceStep, sym::ExprBinaryOp::Mul, *sliceRowStride);
    if (failed(flatStepExpr))
      return failure();

    // Build the symbol-binding map the same way `composeAccessBaseOffset`
    // does for the per-access patterns: the source operand's 1-to-N
    // expansion supplies dim / stride aux; idx-typed subscripts bind
    // their own bare symbol; ancestor block args (loop induction vars)
    // bind any bare-sym `!hc.idx` they carry. `materializeOffsetSSA`
    // emits `hc.idx_apply` ops with the right operand list so the
    // launch-body lowering downstream picks them up by SSA, not by
    // ambient resolution.
    llvm::StringMap<Value> bindings;
    SmallVector<std::string> implicitSyms = collectImplicitSyms(preFlattenSrc);
    if (sourceAux.size() == implicitSyms.size())
      for (auto [name, value] : llvm::zip_equal(implicitSyms, sourceAux))
        bindings[name] = value;
    auto pinsBareSymbol = [&store](Type type) -> StringRef {
      auto idxType = dyn_cast<IdxType>(type);
      if (!idxType)
        return {};
      ExprAttr exprAttr = idxType.getExpr();
      if (!exprAttr)
        return {};
      StringRef onlyName;
      bool unique = true;
      sym::walkSymbolNames(exprAttr.getValue(), [&](StringRef name) {
        if (onlyName.empty())
          onlyName = name;
        else if (onlyName != name)
          unique = false;
      });
      if (!unique || onlyName.empty())
        return {};
      auto pinned = sym::composeExprSym(store, onlyName);
      if (failed(pinned))
        return {};
      if (pinned->raw() != exprAttr.getValue().raw())
        return {};
      return onlyName;
    };
    for (Value idx : op.getIndices()) {
      StringRef name = pinsBareSymbol(idx.getType());
      if (name.empty())
        continue;
      bindings.try_emplace(name, idx);
    }
    for (Block *block = op->getBlock(); block;) {
      for (BlockArgument arg : block->getArguments()) {
        StringRef name = pinsBareSymbol(arg.getType());
        if (name.empty())
          continue;
        bindings.try_emplace(name, arg);
      }
      Operation *parent = block->getParentOp();
      if (!parent)
        break;
      block = parent->getBlock();
    }

    auto buildIdx = [&](sym::ExprHandle e) -> Value {
      return materializeOffsetSSA(rewriter, op.getLoc(), ExprAttr::get(ctx, e),
                                  bindings);
    };
    Value newLower = buildIdx(*flatLowerExpr);
    Value newUpper = buildIdx(*flatUpperExpr);
    Value newStep = buildIdx(*flatStepExpr);

    auto sliceType = SliceType::get(ctx, newLower.getType(), newUpper.getType(),
                                    newStep.getType());
    Value newSlice = HCSliceExprOp::create(rewriter, op.getLoc(), sliceType,
                                           newLower, newUpper, newStep)
                         .getResult();

    Value newView =
        HCBufferViewOp::create(rewriter, op.getLoc(), flatResultType,
                               flatSource, ValueRange{newSlice})
            .getResult();

    return pushReplacement(newView);
  }
};

// Compose the per-operand per-axis offset arrays on `hc.generic` into
// single-entry arrays — the post-flatten contract per `doc/layouts.md`.
// For each shaped operand the rewrite reads the operand's pre-flatten
// layout / shape, treats the per-axis array as the access-site index
// list, and composes a single 1D offset through ixsimpl using the same
// machinery the per-access patterns above use. Without an explicit
// layout the fallback is the identity layout over the operand's shape.
//
// Free symbols of the composed offset (iter syms, dim / stride params)
// stay free — they're resolved by the surrounding kernel scope and the
// downstream `hc-lower-generic` lowering. Operands that aren't shaped
// (`!hc.ptr<...>`, `!hc.undef`) keep their offset arrays unchanged: a
// ptr already rides on a single 1D address by ODS contract and undef
// has no shape to compose against.
//
// The pattern also performs the type-only retype that `RetypeAnyHCOp`
// would otherwise have done — the operand 1-to-N expansion is sliced
// to its leading flat carrier per operand, result types run through
// the converter, and trailing aux values flow into the result
// expansions via the shared sym-name binding map. Doing both here
// keeps the post-flatten op single-pass: nD offset arrays never live
// alongside 1D operand types in the IR.
//
// On any rank mismatch / missing layout (buffer with no layout) the
// rewrite bails. The verifier on `hc.generic` enforces
// `len(offset array) == operand rank`, so a bailed op surfaces as a
// downstream verification error rather than silent miscompile.
struct ComposeGenericOffsets : public ComposeAccessOffsetBase<HCGenericOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCGenericOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCGenericOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    ArrayAttr insOffsets = op.getInsOffsetsAttr();
    ArrayAttr outsOffsets = op.getOutsOffsetsAttr();

    // Compose each per-operand offset array. Operands that aren't
    // shaped (or are already-flat) pass through unchanged. Returns a
    // failure to bail the whole rewrite to RetypeAnyHCOp; returns
    // `false` to indicate "nothing changed for this operand".
    bool composed = false;
    auto composeOne = [&](Value origOperand,
                          ArrayAttr perAxis) -> FailureOr<ArrayAttr> {
      Type t = origOperand.getType();
      auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
      if (!shaped)
        return perAxis;
      ShapeAttr shape = shaped.getSymbolicShape();
      if (!shape)
        return perAxis;
      LayoutAttr layout = shaped.getSymbolicLayout();
      // Layout-less operands (including buffers that haven't picked
      // up the default strided layout — `hc-canonicalize-layouts`
      // only attaches one for kernel-arg buffers) fall back to the
      // identity layout, the canonical contract for layout-free
      // shaped types. Same path `composeAccessOffsetExpr` takes for
      // tensor / vector operands.
      // Rank-0 has nothing to compose; ditto for an op produced with
      // an empty offset array on the operand.
      if (perAxis.empty())
        return perAxis;

      SmallVector<ExprAttr> perAxisExprs;
      perAxisExprs.reserve(perAxis.size());
      for (Attribute a : perAxis) {
        auto expr = dyn_cast<ExprAttr>(a);
        if (!expr)
          return failure();
        perAxisExprs.push_back(expr);
      }
      auto offset = composeAccessOffsetExpr(ctx, layout, shape, perAxisExprs);
      if (failed(offset))
        return failure();
      auto result = ArrayAttr::get(ctx, ArrayRef<Attribute>{*offset});
      // A rank-1 layout-less operand round-trips through the identity
      // layout to itself; don't flip `composed` for an unchanged
      // attribute or the pattern would claim a rewrite happened when
      // nothing on the surface moved.
      if (result != perAxis)
        composed = true;
      return result;
    };

    SmallVector<Attribute> newInsOffsets;
    newInsOffsets.reserve(insOffsets.size());
    for (auto [in, perAxis] :
         llvm::zip_equal(op.getIns(), insOffsets.getAsRange<ArrayAttr>())) {
      auto out = composeOne(in, perAxis);
      if (failed(out))
        return failure();
      newInsOffsets.push_back(*out);
    }

    SmallVector<Attribute> newOutsOffsets;
    newOutsOffsets.reserve(outsOffsets.size());
    for (auto [out, perAxis] :
         llvm::zip_equal(op.getOuts(), outsOffsets.getAsRange<ArrayAttr>())) {
      auto composedAttr = composeOne(out, perAxis);
      if (failed(composedAttr))
        return failure();
      newOutsOffsets.push_back(*composedAttr);
    }

    // Flat carrier slicing matches RetypeAnyHCOp's contract — each
    // 1-to-N adapter range hands back the flat shaped value as the
    // leading entry; trailing aux values feed the result-expansion
    // binding map.
    if (adaptor.getIterBounds().size() != op.getIterBounds().size())
      return failure();
    SmallVector<Value> iterBounds;
    iterBounds.reserve(adaptor.getIterBounds().size());
    for (ValueRange range : adaptor.getIterBounds()) {
      if (range.size() != 1)
        return failure();
      iterBounds.push_back(range.front());
    }

    SmallVector<Value> flatIns;
    flatIns.reserve(adaptor.getIns().size());
    for (ValueRange range : adaptor.getIns()) {
      if (range.empty())
        return failure();
      flatIns.push_back(range.front());
    }
    SmallVector<Value> flatOuts;
    flatOuts.reserve(adaptor.getOuts().size());
    for (ValueRange range : adaptor.getOuts()) {
      if (range.empty())
        return failure();
      flatOuts.push_back(range.front());
    }

    // Pull the shape-preserving sym-name bindings off every operand
    // expansion so result aux can be sourced from a matching name
    // before falling back to an ambient `hc.idx_apply`.
    llvm::StringMap<Value> bindings;
    for (auto [orig, range] : llvm::zip_equal(op.getIns(), adaptor.getIns()))
      noteOperandBindings(orig.getType(), range, bindings);
    for (auto [orig, range] : llvm::zip_equal(op.getOuts(), adaptor.getOuts()))
      noteOperandBindings(orig.getType(), range, bindings);

    // Capture ambient sym → SSA bindings now, while the kernel-arg
    // bundle UCC chain, gpu.launch block args, and structured-loop
    // induction vars are all still HC-typed and reachable. Walking
    // ancestor blocks here also picks up `$joinN` from `hc.for_range`'s
    // IV — `hc-lower-launch-body` will later rewrite for_range to
    // scf.for and strip the `!hc.idx<sym>` payload off the IV, but
    // by then `ambient_idxs` already holds the SSA edge and the
    // launch-body type converter only changes the operand's type
    // (the sym name lives on `ambient_idx_syms`).
    auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
    collectAncestorIdxBindings(store, op, bindings);

    // Collect every free symbol that the composed offset expressions
    // reference, minus the iter syms (those are the per-iteration
    // axes, scoped to the hc.generic body — they remain free and are
    // substituted per-lane by `hc-lower-generic`).
    llvm::StringSet<> iterSymSet;
    for (Attribute s : op.getIterSymsAttr())
      iterSymSet.insert(cast<StringAttr>(s).getValue());
    llvm::StringSet<> ambientNeeded;
    auto walkOffsets = [&](ArrayRef<Attribute> perOperandAttrs) {
      for (Attribute a : perOperandAttrs) {
        auto perOperand = dyn_cast<ArrayAttr>(a);
        if (!perOperand)
          continue;
        for (Attribute axisAttr : perOperand) {
          auto expr = dyn_cast<ExprAttr>(axisAttr);
          if (!expr)
            continue;
          sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
            if (!iterSymSet.contains(name))
              ambientNeeded.insert(name);
          });
        }
      }
    };
    walkOffsets(newInsOffsets);
    walkOffsets(newOutsOffsets);
    // Body `hc.idx_apply` / `hc.pred_apply` ops can reference ambient
    // syms directly (post-flatten, the body authoring convention is
    // that any sym not on the apply's `symbols` list is either an iter
    // sym scoped to the body or an ambient sym the surrounding
    // `hc.generic` is responsible for plumbing). Pick those up too so
    // `hc-lower-generic` can seed the per-lane scope from
    // `ambient_idxs` and the second `hc-lower-launch-body` invocation
    // finds bindings for them via `seedAmbientScope` inside the
    // unrolled body. The `hc.load_mask` rewrite in
    // `hc-load-store-to-generic` is today the only emitter that puts
    // ambient-referencing applies inside the body (its predicate is
    // `(lo + step*i_k) < D_k` with `lo` / `D_k` being kernel-arg or
    // launch-geometry syms) — extending the walk now keeps the
    // contract general.
    auto collectFreeNames =
        [&](ArrayAttr existing,
            llvm::function_ref<void(llvm::function_ref<void(StringRef)>)>
                walker) {
          llvm::StringSet<> already;
          for (Attribute n : existing)
            already.insert(cast<StringAttr>(n).getValue());
          walker([&](StringRef name) {
            if (already.contains(name))
              return;
            if (iterSymSet.contains(name))
              return;
            ambientNeeded.insert(name);
          });
        };
    auto walkBodyApplies = [&](Block &body) {
      for (Operation &nested : body) {
        if (auto idx = dyn_cast<HCIdxApplyOp>(&nested)) {
          auto idxTy = dyn_cast<IdxType>(idx.getResult().getType());
          if (idxTy && idxTy.getExpr())
            collectFreeNames(idx.getSymbolsAttr(),
                             [&](llvm::function_ref<void(StringRef)> cb) {
                               sym::walkSymbolNames(idxTy.getExpr().getValue(),
                                                    cb);
                             });
        } else if (auto predOp = dyn_cast<HCPredApplyOp>(&nested)) {
          auto predTy = dyn_cast<PredType>(predOp.getResult().getType());
          if (predTy && predTy.getPred())
            collectFreeNames(predOp.getSymbolsAttr(),
                             [&](llvm::function_ref<void(StringRef)> cb) {
                               sym::walkSymbolNames(predTy.getPred().getValue(),
                                                    cb);
                             });
        }
      }
    };
    if (!op.getBody().empty())
      walkBodyApplies(op.getBody().front());
    // Carry over any ambient bindings the source op already had — the
    // pre-flatten emitters may have left them empty, but if a prior
    // pass populated them we don't want to drop the SSA edge silently.
    for (auto [val, symAttr] :
         llvm::zip_equal(adaptor.getAmbientIdxs(),
                         op.getAmbientIdxSymsAttr().getAsRange<StringAttr>())) {
      // adaptor handed us a per-operand value range — pick the first
      // entry (the operand itself; trailing aux belongs to its own
      // expansion).
      if (val.empty())
        continue;
      bindings.try_emplace(symAttr.getValue(), val.front());
      ambientNeeded.insert(symAttr.getValue());
    }
    // Lex-sort for deterministic operand order. Pin bindings whose
    // SSA we resolved; leave the rest to ambient-context resolution
    // downstream (the symbol stays free in the offset expression and
    // `hc.idx_apply`'s severing form handles it).
    SmallVector<StringRef> ambientNames(ambientNeeded.keys().begin(),
                                        ambientNeeded.keys().end());
    llvm::sort(ambientNames);
    SmallVector<Value> ambientIdxsVec;
    SmallVector<Attribute> ambientSymsVec;
    for (StringRef name : ambientNames) {
      auto it = bindings.find(name);
      if (it == bindings.end())
        continue;
      ambientIdxsVec.push_back(it->second);
      ambientSymsVec.push_back(rewriter.getStringAttr(name));
    }

    // Convert result types via the 1-to-N converter. The new op only
    // carries the leading flat type per result; trailing aux values
    // are SSA-generated alongside.
    SmallVector<Type> flatResultTypes;
    SmallVector<unsigned> resultWidths;
    flatResultTypes.reserve(op.getNumResults());
    resultWidths.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      SmallVector<Type> converted;
      if (failed(getTypeConverter()->convertType(resultType, converted)))
        return failure();
      if (converted.empty())
        return failure();
      flatResultTypes.push_back(converted.front());
      resultWidths.push_back(converted.size());
    }

    // Bail if neither offsets nor types changed — failure here lets
    // the driver short-circuit to the next pattern instead of looping
    // on a no-op rewrite.
    bool typesChanged = false;
    for (auto [oldT, newT] :
         llvm::zip_equal(op.getResultTypes(), flatResultTypes))
      if (oldT != newT) {
        typesChanged = true;
        break;
      }
    if (!typesChanged)
      for (auto [oldVal, range] :
           llvm::zip(op.getOperands(), adaptor.getOperands()))
        if (range.size() != 1 || oldVal.getType() != range.front().getType()) {
          typesChanged = true;
          break;
        }
    if (!composed && !typesChanged)
      return failure();

    auto newOp = HCGenericOp::create(
        rewriter, op.getLoc(), flatResultTypes, op.getIterSymsAttr(),
        ValueRange(iterBounds), op.getIterKindsAttr(), ValueRange(flatIns),
        ValueRange(flatOuts), /*ambient_idxs=*/ValueRange(ambientIdxsVec),
        rewriter.getArrayAttr(ambientSymsVec),
        ArrayAttr::get(ctx, newInsOffsets),
        ArrayAttr::get(ctx, newOutsOffsets));
    // Carry over any extra discardable attributes (e.g. location-name
    // hints) the original op picked up before we got here. The named
    // attributes the builder wrote — iter_syms, iter_kinds, ins/outs
    // offsets, segment sizes — already match.
    StringSet<> handled = {
        op.getIterSymsAttrName().getValue(),
        op.getIterKindsAttrName().getValue(),
        op.getAmbientIdxSymsAttrName().getValue(),
        op.getInsOffsetsAttrName().getValue(),
        op.getOutsOffsetsAttrName().getValue(),
        op.getOperandSegmentSizesAttrName().getValue(),
    };
    for (NamedAttribute attr : op->getAttrs())
      if (!handled.contains(attr.getName().getValue()))
        newOp->setAttr(attr.getName(), attr.getValue());

    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(
            rewriter.convertRegionTypes(&newOp.getBody(), *getTypeConverter())))
      return failure();

    SmallVector<SmallVector<Value>> replacementStorage;
    replacementStorage.reserve(op.getNumResults());
    SmallVector<ValueRange> replacements;
    replacements.reserve(op.getNumResults());
    for (auto [newResult, origResult] :
         llvm::zip_equal(newOp.getResults(), op.getResults())) {
      auto aux = resolveResultAuxValues(rewriter, op.getLoc(),
                                        origResult.getType(), bindings);
      if (failed(aux))
        return failure();
      SmallVector<Value> bundle = {newResult};
      llvm::append_range(bundle, *aux);
      replacementStorage.push_back(std::move(bundle));
      replacements.push_back(replacementStorage.back());
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

// Generic op-rebuild pattern for HC dialect ops. The function/SCF
// populators retype signatures and structural ops, but ops in the
// middle of the IR (`hc.generic`, `hc.load`, `hc.cast`, ...) need to
// be rebuilt with converted operand/result types so the post-flatten
// IR carries 1D types end-to-end without `unrealized_conversion_cast`
// stranded in the middle. This pattern is "type-only" by design —
// attributes (including the per-axis `#hc.expr` offset arrays on
// `hc.generic`) and regions are carried over unchanged. Body block
// args of `hc.generic` and friends carry scalar element types that
// don't change under the flatten, so no signature conversion is
// needed inside the regions.
//
// Under the 1-to-N converter the rewrite hands the new op only the
// leading flat carrier of each 1-to-N operand expansion; the
// trailing aux idx values flow into the result expansions via the
// shared sym-name binding map (shape-preserving ops by default
// thread the same aux through, since the result and operand share
// the same implicit sym set). When a result aux can't be sourced
// from any operand the helper emits an empty-binding `hc.idx_apply`
// — same severance form `materialize-bound-exprs` lays down — so
// the launch-body lowering's ambient walker can resolve it.
//
// Scoped to ops in the `hc` dialect to stay out of the way of
// upstream patterns (func / scf / call) and avoid silent wins over
// patterns the rest of the codebase relies on. Returning `failure()`
// on a no-op match lets the driver short-circuit to the next pattern
// without us paying for a clone.
struct RetypeAnyHCOp : public ConversionPattern {
  RetypeAnyHCOp(const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getDialect() != op->getContext()->getLoadedDialect<HCDialect>())
      return failure();

    // Compute the 1-to-N converted result types.
    SmallVector<Type> convertedResultTypes;
    SmallVector<unsigned> resultWidths;
    if (failed(convertResultTypes(op->getResultTypes(), *getTypeConverter(),
                                  convertedResultTypes, resultWidths)))
      return failure();

    // Cheap no-op check: skip if every result and every operand is
    // already legal under the converter (the driver would re-fire us
    // forever otherwise).
    bool changed = false;
    if (op->getNumResults() != convertedResultTypes.size())
      changed = true;
    if (!changed)
      for (auto [oldT, newT] :
           llvm::zip_equal(op->getResultTypes(), convertedResultTypes))
        if (oldT != newT) {
          changed = true;
          break;
        }
    if (!changed) {
      for (auto [oldVal, range] :
           llvm::zip_equal(op->getOperands(), operands)) {
        if (range.size() != 1 || oldVal.getType() != range.front().getType()) {
          changed = true;
          break;
        }
      }
    }
    if (!changed)
      return failure();

    // Slice the leading flat values out of every 1-to-N operand;
    // the new op only sees those (the trailing aux carries the dim
    // and stride values that this op doesn't consume).
    SmallVector<Value> flatOperands = flatOperandsOnly(operands);

    // Build a per-name binding map from the operand expansions so
    // result aux can be sourced from a matching operand sym instead
    // of a fresh ambient apply.
    llvm::StringMap<Value> bindings;
    for (auto [origOperand, range] :
         llvm::zip_equal(op->getOperands(), operands))
      noteOperandBindings(origOperand.getType(), range, bindings);

    // Fresh op with flat-only operands and flat-only result types
    // (the aux idx values are SSA-generated alongside, not on the
    // op surface).
    SmallVector<Type> flatResultTypes;
    flatResultTypes.reserve(op->getNumResults());
    unsigned offset = 0;
    for (unsigned width : resultWidths) {
      if (width == 0)
        return failure();
      flatResultTypes.push_back(convertedResultTypes[offset]);
      offset += width;
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(flatOperands);
    state.addTypes(flatResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (size_t i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();
    Operation *newOp = rewriter.create(state);
    for (auto [oldRegion, newRegion] :
         llvm::zip_equal(op->getRegions(), newOp->getRegions()))
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());

    // Block arguments inside the inlined regions still carry the
    // pre-flatten types — `inlineRegionBefore` doesn't run the
    // converter on them. `hc.for_range` is the canonical victim:
    // its body's iter-arg block args must match the converted
    // `iter_inits` operand types or the parent verifier rejects the
    // op. Run the converter across every region; for ops whose body
    // block args don't get flattened (`hc.generic`'s scalar element
    // types, etc.) this is a no-op.
    for (Region &region : newOp->getRegions()) {
      if (region.empty())
        continue;
      if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter())))
        return failure();
    }

    // Pair each new flat result with its aux values for replacement.
    SmallVector<SmallVector<Value>> replacementStorage;
    replacementStorage.reserve(op->getNumResults());
    SmallVector<ValueRange> replacements;
    replacements.reserve(op->getNumResults());
    for (auto [newResult, origResult] :
         llvm::zip_equal(newOp->getResults(), op->getResults())) {
      auto aux = resolveResultAuxValues(rewriter, op->getLoc(),
                                        origResult.getType(), bindings);
      if (failed(aux))
        return failure();
      SmallVector<Value> bundle = {newResult};
      llvm::append_range(bundle, *aux);
      replacementStorage.push_back(std::move(bundle));
      replacements.push_back(replacementStorage.back());
    }

    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }

  // Local ResultTypes helper that reuses the 1-to-N converter to
  // build (per-result widths + flat type list).
  static LogicalResult
  convertResultTypes(TypeRange resultTypes, const TypeConverter &converter,
                     SmallVectorImpl<Type> &convertedResults,
                     SmallVectorImpl<unsigned> &resultWidths) {
    for (Type resultType : resultTypes) {
      unsigned start = convertedResults.size();
      if (failed(converter.convertType(resultType, convertedResults)))
        return failure();
      resultWidths.push_back(convertedResults.size() - start);
    }
    return success();
  }
};

// Build a sparse `DictionaryAttr` keyed by stringified post-flatten arg
// index. Each entry records, for one of flatten's aux `!hc.idx<sym>`
// slots, the parent (flat carrier) buffer arg's post-flatten index, the
// axis the aux corresponds to in the parent's pre-flatten shape, and
// whether it's a `"dim"` or `"stride"` aux. Pre-flatten the host
// wrapper recovers dim values from the buffer arg's symbolic shape;
// post-flatten the shape collapses to `[?]` and the per-axis info
// lives on the trailing aux slots — the meta lets the lowering pair
// each aux slot back to a `(parent buf pyArg, axis, kind)` triple
// without re-deriving it from the surrounding signature.
//
// Returns a null attribute when no buffer arg expanded (e.g. a func
// that only takes scalars, or one whose buffers were already flat).
static DictionaryAttr buildFlattenAuxArgsMeta(MLIRContext *ctx,
                                              FunctionType origType,
                                              const TypeConverter &converter) {
  SmallVector<NamedAttribute> entries;
  auto i64Type = IntegerType::get(ctx, 64);
  unsigned newIdx = 0;
  for (Type origInput : origType.getInputs()) {
    SmallVector<Type> converted;
    if (failed(converter.convertType(origInput, converted))) {
      ++newIdx;
      continue;
    }

    auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(origInput);
    auto buffer = dyn_cast<BufferType>(origInput);
    if (!buffer || !shaped || isAlreadyFlat(shaped)) {
      newIdx += converted.size();
      continue;
    }

    SmallVector<StringRef> axisSyms;
    if (ShapeAttr shape = shaped.getSymbolicShape())
      for (Attribute dim : shape.getDims())
        axisSyms.push_back(bareDimSymbolName(dim));

    SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
    unsigned flatCarrierIdx = newIdx;

    for (auto [i, sym] : llvm::enumerate(implicitSyms)) {
      unsigned auxPostIdx = flatCarrierIdx + 1 + i;
      StringRef name(sym);
      StringRef kind;
      int64_t axis = -1;
      if (std::optional<unsigned> strideAxis = parseStrideAxis(name)) {
        kind = "stride";
        axis = static_cast<int64_t>(*strideAxis);
      } else {
        kind = "dim";
        for (auto [a, axisSym] : llvm::enumerate(axisSyms))
          if (axisSym == name) {
            axis = static_cast<int64_t>(a);
            break;
          }
      }
      // A dim aux with no matching shape axis would mean the implicit
      // sym leaked in via a non-shape source (`storage_size`, layout
      // params); the host wrapper can't bind it from `hc_get_dim`
      // alone, so drop the entry rather than emit an ambiguous one.
      if (kind == "dim" && axis < 0)
        continue;

      SmallVector<NamedAttribute> auxEntries;
      auxEntries.emplace_back(StringAttr::get(ctx, "aux_of"),
                              IntegerAttr::get(i64Type, flatCarrierIdx));
      auxEntries.emplace_back(StringAttr::get(ctx, "axis"),
                              IntegerAttr::get(i64Type, axis));
      auxEntries.emplace_back(StringAttr::get(ctx, "kind"),
                              StringAttr::get(ctx, kind));

      SmallString<8> key;
      Twine(auxPostIdx).toVector(key);
      entries.emplace_back(StringAttr::get(ctx, key),
                           DictionaryAttr::get(ctx, auxEntries));
    }

    newIdx += converted.size();
  }

  if (entries.empty())
    return {};
  return DictionaryAttr::get(ctx, entries);
}

// Update the `function_type` attribute and body block arguments of a
// HC dialect symbol op (`hc.intrinsic`, `hc.func`, `hc.kernel`) so the
// signature stays in sync with the converted call sites and bodies.
// Unlike `func.func`, these ops don't implement `FunctionOpInterface`,
// so the upstream populator doesn't see them; without this pattern,
// the verifier on `hc.call_intrinsic` / `hc.call` rejects the op once
// the operand types diverge from the still-original declared
// signature.
//
// The body block args are converted via `applySignatureConversion` so
// the 1-to-N expansion the converter emits for shaped types (one flat
// carrier + N aux `!hc.idx<sym>` for free dim/stride symbols) lands
// as parallel block arguments — the same surface call sites get on
// their operand expansion.
//
// `hc.flatten_aux_args` is attached on rewrite so the host-wrapper
// lowering (`hc-lower-kernels-to-gpu-launch`) can identify which
// post-flatten args are flatten-emitted aux slots (vs user-passed
// scalars) and resolve their values from the parent buffer's PyObject
// instead of allocating a fresh PyObject slot per aux. See
// `buildFlattenAuxArgsMeta` for the attribute shape.
template <typename SymbolOp>
struct ConvertHCSymbolSignatureOp : public OpConversionPattern<SymbolOp> {
  using OpConversionPattern<SymbolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SymbolOp op,
                  typename OpConversionPattern<SymbolOp>::OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<FunctionType> fnType = op.getFunctionType();
    if (!fnType)
      return failure();

    SmallVector<Type> ins;
    if (failed(
            this->getTypeConverter()->convertTypes(fnType->getInputs(), ins)))
      return failure();
    SmallVector<Type> outs;
    if (failed(
            this->getTypeConverter()->convertTypes(fnType->getResults(), outs)))
      return failure();
    auto newType = FunctionType::get(rewriter.getContext(), ins, outs);
    if (newType == *fnType)
      return failure();

    DictionaryAttr auxMeta = buildFlattenAuxArgsMeta(
        rewriter.getContext(), *fnType, *this->getTypeConverter());

    Region &body = op.getBody();
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(newType);
      if (auxMeta)
        op->setAttr("hc.flatten_aux_args", auxMeta);
      if (body.empty())
        return;
      TypeConverter::SignatureConversion conversion(fnType->getNumInputs());
      for (auto [index, input] : llvm::enumerate(fnType->getInputs())) {
        SmallVector<Type> converted;
        if (failed(this->getTypeConverter()->convertType(input, converted)))
          return;
        conversion.addInputs(index, converted);
      }
      rewriter.applySignatureConversion(&body.front(), conversion,
                                        this->getTypeConverter());
    });
    return success();
  }
};

struct HCFlattenWithLayoutsPass final
    : public hc::impl::HCFlattenWithLayoutsBase<HCFlattenWithLayoutsPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    FlattenLayoutConverter converter;

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    // Per-access-op patterns are listed first by intent — the
    // conversion driver still picks via benefit (2 vs the generic
    // retype's 1), but having them grouped reads as the design.
    patterns.add<ComposeLoadOffsets, ComposeVLoadOffsets, ComposeStoreOffsets,
                 ComposeGenericOffsets, ComposeBufferViewOffsets, DropAsLayout,
                 RetypeAnyHCOp, ConvertHCSymbolSignatureOp<HCIntrinsicOp>,
                 ConvertHCSymbolSignatureOp<HCFuncOp>,
                 ConvertHCSymbolSignatureOp<HCKernelOp>>(converter, ctx);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto fn = dyn_cast<FunctionOpInterface>(op))
        if (auto fnType = dyn_cast<FunctionType>(fn.getFunctionType()))
          return converter.isSignatureLegal(fnType);
      if (isa<func::ReturnOp, func::CallOp>(op))
        return converter.isLegal(op);
      // `hc.as_layout` is always illegal: the rewriter above unconditionally
      // drops it. Without that gate the driver would consider the op legal
      // when both endpoints already have the same converted type, leaving
      // the cosmetic relabel in place.
      if (isa<HCAsLayoutOp>(op))
        return false;
      // HC's symbol-carrying ops carry their signature in a
      // `function_type` attribute; `converter.isLegal(op)` only inspects
      // operand/result types, which are zero on these ops. Match the
      // upstream `FunctionOpInterface` legality rule explicitly.
      auto checkSignature = [&](FunctionType fnType) {
        return converter.isSignatureLegal(fnType);
      };
      if (auto kernel = dyn_cast<HCKernelOp>(op))
        if (auto fnType = kernel.getFunctionType())
          return checkSignature(*fnType);
      if (auto fn = dyn_cast<HCFuncOp>(op))
        if (auto fnType = fn.getFunctionType())
          return checkSignature(*fnType);
      if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op))
        if (auto fnType = intrinsic.getFunctionType())
          return checkSignature(*fnType);
      // `hc.generic` legality also gates on offset-array rank parity:
      // a type-only retype with no offset composition leaves nD arrays
      // sitting on now-1D operands, which the post-flatten verifier
      // rejects. Force the driver back through `ComposeGenericOffsets`
      // when any operand's offset count diverges from its rank.
      if (auto generic = dyn_cast<HCGenericOp>(op)) {
        if (!converter.isLegal(op))
          return false;
        auto operandRank = [](Type t) -> std::optional<size_t> {
          if (isHCUndefType(t))
            return std::nullopt;
          if (isa<PtrType>(t))
            return size_t{1};
          if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t))
            if (ShapeAttr shape = shaped.getSymbolicShape())
              return shape.getDims().size();
          return std::nullopt;
        };
        auto checkRoleParity = [&](OperandRange ops, ArrayAttr offsets) {
          for (auto [val, off] :
               llvm::zip_equal(ops, offsets.getAsRange<ArrayAttr>())) {
            std::optional<size_t> rank = operandRank(val.getType());
            if (rank && off.size() != *rank)
              return false;
          }
          return true;
        };
        if (!checkRoleParity(generic.getIns(), generic.getInsOffsetsAttr()))
          return false;
        if (!checkRoleParity(generic.getOuts(), generic.getOutsOffsetsAttr()))
          return false;
        return true;
      }
      // HC dialect ops are legal iff every operand and every result type
      // is already in its converted form. The `RetypeAnyHCOp` pattern
      // takes care of the rebuild when one side still carries the
      // pre-flatten shape.
      if (op->getDialect() == ctx->getLoadedDialect<HCDialect>())
        return converter.isLegal(op);
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
