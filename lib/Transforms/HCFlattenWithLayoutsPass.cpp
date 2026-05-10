// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-flatten-with-layouts`. Every shaped value loses its
// `#hc.layout` slot AND collapses its shape to a single entry. Tensors
// and vectors get a concrete `storage_size_expr` (from the layout's
// `storage_size` after binding `shape_syms` to the original shape, or
// the dim product when the implicit identity-row-major contract
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
// The op-level structural invariants stay nD: `hc.generic` keeps its
// per-axis offset arrays at the original logical rank for downstream
// fusion / vectorization, and the rewriter does not touch them.
//
// Per-access ops (`hc.load`, `hc.vload`, `hc.store`, `hc.load_mask`)
// also get rewritten in this pass: their multi-index lists collapse
// to a single 1D base-offset SSA value composed from the operand's
// layout against the access site's index expressions. Composition
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
// implicit identity-row-major contract and the storage size is the
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
// store any other producer of these expressions would.
static FailureOr<ExprAttr> computeStorageSizeExpr(MLIRContext *ctx,
                                                  LayoutAttr layout,
                                                  ShapeAttr originalShape) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

  if (!layout) {
    ArrayRef<Attribute> dims = originalShape.getDims();
    if (dims.empty()) {
      auto one = sym::composeExprInt(store, 1);
      if (failed(one))
        return failure();
      return ExprAttr::get(ctx, *one);
    }
    sym::ExprHandle product = llvm::cast<ExprAttr>(dims[0]).getValue();
    for (Attribute dim : dims.drop_front()) {
      auto next = sym::composeExprBinary(store, product, sym::ExprBinaryOp::Mul,
                                         llvm::cast<ExprAttr>(dim).getValue());
      if (failed(next))
        return failure();
      product = *next;
    }
    return ExprAttr::get(ctx, product);
  }

  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> dims = originalShape.getDims();
  if (shapeSyms.size() != dims.size())
    return failure();

  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(shapeSyms.size());
  replacements.reserve(shapeSyms.size());
  for (auto [sym, dim] : llvm::zip_equal(shapeSyms, dims)) {
    auto symHandle =
        sym::composeExprSym(store, llvm::cast<StringAttr>(sym).getValue());
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(
        const_cast<ixs_node *>(llvm::cast<ExprAttr>(dim).getValue().raw()));
  }

  sym::Session session(store);
  ixs_node *bound = ixs_subs_multi(
      session.raw(),
      const_cast<ixs_node *>(layout.getStorageSize().getValue().raw()),
      static_cast<uint32_t>(targets.size()), targets.data(),
      replacements.data());
  if (!bound)
    return failure();
  return ExprAttr::get(ctx, sym::ExprHandle(bound));
}

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

// Build the identity row-major offset for an access into a layout-less
// shaped operand: `i_0 * (d_1 * ... * d_{n-1}) + i_1 * (d_2 * ... *
// d_{n-1}) + ... + i_{n-1}`. Right-to-left fold gives a single ixsimpl
// pass on the way out, which canonicalizes the result for free.
// Rank-0 returns `0`. Caller checks rank parity.
static FailureOr<sym::ExprHandle>
identityRowMajorOffset(sym::Store &store, ArrayRef<ExprAttr> indexExprs,
                       ArrayRef<Attribute> dims) {
  auto zero = sym::composeExprInt(store, 0);
  if (failed(zero))
    return failure();
  if (indexExprs.empty())
    return *zero;
  sym::ExprHandle accum = *zero;
  for (size_t i = 0; i < indexExprs.size(); ++i) {
    sym::ExprHandle term = indexExprs[i].getValue();
    for (size_t j = i + 1; j < dims.size(); ++j) {
      auto dimExpr = llvm::dyn_cast<ExprAttr>(dims[j]);
      if (!dimExpr)
        return failure();
      auto next = sym::composeExprBinary(store, term, sym::ExprBinaryOp::Mul,
                                         dimExpr.getValue());
      if (failed(next))
        return failure();
      term = *next;
    }
    auto added =
        sym::composeExprBinary(store, accum, sym::ExprBinaryOp::Add, term);
    if (failed(added))
      return failure();
    accum = *added;
  }
  return accum;
}

// Compose the access base offset for a multi-index access into a shaped
// operand. With a layout, substitute `shape_syms` positionally with the
// operand's shape entries and `index_syms` positionally with the access
// site's index expressions, then evaluate `layout.offset`. Without a
// layout, fall back to identity row-major over the operand's shape (the
// canonical contract for layout-less shaped types). Rank parity between
// the operand shape and the index list is the caller's responsibility —
// the verifier on the access op already enforces it pre-rewrite, but
// the helper still bails on mismatch instead of producing nonsense.
static FailureOr<ExprAttr>
composeAccessOffsetExpr(MLIRContext *ctx, LayoutAttr layout,
                        ShapeAttr originalShape,
                        ArrayRef<ExprAttr> indexExprs) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  ArrayRef<Attribute> dims = originalShape.getDims();
  if (dims.size() != indexExprs.size())
    return failure();

  if (!layout) {
    auto offset = identityRowMajorOffset(store, indexExprs, dims);
    if (failed(offset))
      return failure();
    return ExprAttr::get(ctx, *offset);
  }

  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = layout.getIndexSyms();
  if (shapeSyms.size() != dims.size() || indexSyms.size() != indexExprs.size())
    return failure();

  // Single substitution pass over `shape_syms ++ index_syms`. Params
  // intentionally don't expand here — they survive as free symbols in
  // the resulting offset, the same way `computeStorageSizeExpr` lets
  // them survive in the post-flatten storage size. Whoever resolves
  // those symbols downstream (launch context for stride params, the
  // launch-body lowering's idx_apply walk for closed-form ones) does
  // it uniformly across both surfaces.
  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(shapeSyms.size() + indexSyms.size());
  replacements.reserve(shapeSyms.size() + indexSyms.size());
  auto pushPair = [&](StringRef name,
                      sym::ExprHandle replacement) -> LogicalResult {
    auto symHandle = sym::composeExprSym(store, name);
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(const_cast<ixs_node *>(replacement.raw()));
    return success();
  };
  for (auto [sym, dim] : llvm::zip_equal(shapeSyms, dims)) {
    auto dimExpr = llvm::dyn_cast<ExprAttr>(dim);
    if (!dimExpr)
      return failure();
    if (failed(pushPair(llvm::cast<StringAttr>(sym).getValue(),
                        dimExpr.getValue())))
      return failure();
  }
  for (auto [sym, idx] : llvm::zip_equal(indexSyms, indexExprs)) {
    if (failed(
            pushPair(llvm::cast<StringAttr>(sym).getValue(), idx.getValue())))
      return failure();
  }

  sym::Session session(store);
  ixs_node *bound = ixs_subs_multi(
      session.raw(),
      const_cast<ixs_node *>(layout.getOffset().getValue().raw()),
      static_cast<uint32_t>(targets.size()), targets.data(),
      replacements.data());
  if (!bound)
    return failure();
  return ExprAttr::get(ctx, sym::ExprHandle(bound));
}

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
  // here would mean we'd fall back to identity row-major over the
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

  for (Value idx : indices) {
    auto idxType = dyn_cast<IdxType>(idx.getType());
    if (!idxType)
      continue;
    ExprAttr expr = idxType.getExpr();
    if (!expr)
      continue;
    // Only bind when the type pins a *bare* free symbol. Composite
    // expressions (`i + 1`, `i * stride`) are not their own binding
    // for any single name; the value-as-binding shortcut only
    // applies when the type's symbol set is exactly `{name}` and the
    // expression *is* that symbol leaf. The cheapest check: walk and
    // single-out a unique name, then confirm by reconstruction.
    StringRef onlyName;
    bool unique = true;
    sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
      if (onlyName.empty())
        onlyName = name;
      else if (onlyName != name)
        unique = false;
    });
    if (!unique || onlyName.empty())
      continue;
    auto pinned = sym::composeExprSym(store, onlyName);
    if (failed(pinned))
      continue;
    if (pinned->raw() != expr.getValue().raw())
      continue;
    // Don't overwrite a binding from the operand's expansion.
    bindings.try_emplace(onlyName, idx);
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
    if (op.getIndices().size() == 1)
      return failure();
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
                         flatBuffer, ValueRange{*base}, adaptor.getShape()[0]);

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
    if (op.getIndices().size() == 1)
      return failure();
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
                          flatSource, ValueRange{*base}, adaptor.getShape()[0]);

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

struct ComposeLoadMaskOffsets : public ComposeAccessOffsetBase<HCLoadMaskOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCLoadMaskOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCLoadMaskOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIndices().size() == 1)
      return failure();
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

    Type origResultType = op.getMask().getType();
    SmallVector<Type> convertedResults;
    if (failed(
            getTypeConverter()->convertType(origResultType, convertedResults)))
      return failure();
    if (convertedResults.empty())
      return failure();

    auto newLoadMask = HCLoadMaskOp::create(
        rewriter, op.getLoc(), convertedResults.front(), flatSource,
        ValueRange{*base}, adaptor.getShape()[0]);

    llvm::StringMap<Value> bindings;
    noteOperandBindings(op.getSource().getType(), adaptor.getSource(),
                        bindings);
    auto auxValues =
        resolveResultAuxValues(rewriter, op.getLoc(), origResultType, bindings);
    if (failed(auxValues))
      return failure();

    SmallVector<Value> replacement = {newLoadMask.getMask()};
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
    if (op.getIndices().size() == 1)
      return failure();
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
    patterns
        .add<ComposeLoadOffsets, ComposeVLoadOffsets, ComposeLoadMaskOffsets,
             ComposeStoreOffsets, DropAsLayout, RetypeAnyHCOp>(converter, ctx);

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
