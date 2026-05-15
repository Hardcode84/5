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
// Insert every symbol name appearing in `expr` into `seen`. Null
// `expr` is a no-op so callers don't need to special-case missing
// layout pieces (e.g. `LayoutAttr::getOffset()` may legitimately be
// null on a stripped layout).
static void noteExprSymbolNames(ExprAttr expr, llvm::StringSet<> &seen) {
  if (!expr)
    return;
  sym::walkSymbolNames(expr.getValue(),
                       [&](StringRef name) { seen.insert(name); });
}

// Collect every dim-side symbol name referenced by `shaped`.
static void noteSymsInSymbolicShape(SymbolicallyShapedTypeInterface shaped,
                                    llvm::StringSet<> &seen) {
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return;
  for (Attribute dim : shape.getDims())
    if (auto expr = dyn_cast<ExprAttr>(dim))
      noteExprSymbolNames(expr, seen);
}

// Collect every layout-side symbol name referenced by `shaped` (offset,
// storage size, and named layout params), then strip the names that
// bind at access sites (`index_syms`) or that alias dim entries
// (`shape_syms`) — the dim walk already noted those names and we don't
// want both (e.g. `d0` *and* `M`) in the expansion.
static void noteSymsInSymbolicLayout(SymbolicallyShapedTypeInterface shaped,
                                     llvm::StringSet<> &seen) {
  LayoutAttr layout = shaped.getSymbolicLayout();
  if (!layout)
    return;
  noteExprSymbolNames(layout.getOffset(), seen);
  noteExprSymbolNames(layout.getStorageSize(), seen);
  if (DictionaryAttr params = layout.getParams())
    for (NamedAttribute entry : params.getValue())
      if (auto expr = dyn_cast<ExprAttr>(entry.getValue()))
        noteExprSymbolNames(expr, seen);
  for (Attribute attr : layout.getIndexSyms())
    seen.erase(cast<StringAttr>(attr).getValue());
  for (Attribute attr : layout.getShapeSyms())
    seen.erase(cast<StringAttr>(attr).getValue());
}

// table. Sort lexicographically for determinism — `StringSet`
// iteration is unordered and the test corpus pins the operand list
// in textual IR.
static SmallVector<std::string>
collectImplicitSyms(SymbolicallyShapedTypeInterface shaped) {
  llvm::StringSet<> seen;
  noteSymsInSymbolicShape(shaped, seen);
  noteSymsInSymbolicLayout(shaped, seen);

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
// Validate that `preFlattenOperand` is a shaped type carrying a usable
// symbolic shape (and, for buffer operands, a layout). Returns the
// triple `(shaped, layout, shape)` on success. A missing layout on a
// buffer is a frontend bug — every buffer carries the default strided
// layout — and we surface it as a rewrite failure instead of silently
// emitting `0` from an identity-layout fallback over the wrong dims.
struct ShapedAccessOperandInfo {
  SymbolicallyShapedTypeInterface shaped;
  LayoutAttr layout;
  ShapeAttr shape;
};
static FailureOr<ShapedAccessOperandInfo>
validateShapedAccessOperand(Value preFlattenOperand) {
  auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(
      preFlattenOperand.getType());
  if (!shaped)
    return failure();
  ShapeAttr originalShape = shaped.getSymbolicShape();
  if (!originalShape)
    return failure();
  LayoutAttr layout = shaped.getSymbolicLayout();
  if (!layout && llvm::isa<BufferType>(preFlattenOperand.getType()))
    return failure();
  for (Attribute dim : originalShape.getDims())
    if (!llvm::isa<ExprAttr>(dim))
      return failure();
  return ShapedAccessOperandInfo{shaped, layout, originalShape};
}

// Convert each index value's type into an `ExprAttr` via the access
// helper. Any failure short-circuits the whole compose.
static FailureOr<SmallVector<ExprAttr>>
buildAccessIndexExprs(MLIRContext *ctx, sym::Store &store,
                      OperandRange indices) {
  SmallVector<ExprAttr> indexExprs;
  indexExprs.reserve(indices.size());
  for (Value idx : indices) {
    auto expr = extractAccessIndexExpr(ctx, store, idx.getType());
    if (failed(expr))
      return failure();
    indexExprs.push_back(*expr);
  }
  return indexExprs;
}

// Bind one SSA value per implicit symbol from the shaped operand's
// post-flatten aux expansion. The converter is supposed to produce
// the operand and exactly one aux per implicit sym; a mismatch means
// someone fed us a partially-converted operand range and we bail
// rather than guess.
static LogicalResult
bindShapedAuxImplicitSyms(SymbolicallyShapedTypeInterface shaped,
                          ValueRange shapedAux,
                          llvm::StringMap<Value> &bindings) {
  SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
  if (shapedAux.size() != implicitSyms.size())
    return failure();
  for (auto [name, value] : llvm::zip_equal(implicitSyms, shapedAux))
    bindings[name] = value;
  return success();
}

// Pull the bare sym name pinned by an `!hc.idx<sym>` type. Composite
// expressions (`i + 1`, `i * stride`) are not their own binding for
// any single name; the value-as-binding shortcut only applies when
// the type's symbol set is exactly `{name}` and the expression *is*
// that symbol leaf. The cheapest check: walk and single-out a unique
// name, then confirm by reconstruction.
static StringRef pinsBareIdxSymbol(Type type, sym::Store &store) {
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

// Bind each idx-typed access index whose own type pins a single bare
// symbol. Doesn't overwrite a binding already produced by the
// operand's aux expansion.
static void bindBareSymbolIndexOperands(OperandRange indices, sym::Store &store,
                                        llvm::StringMap<Value> &bindings) {
  for (Value idx : indices) {
    StringRef name = pinsBareIdxSymbol(idx.getType(), store);
    if (name.empty())
      continue;
    bindings.try_emplace(name, idx);
  }
}

// Walk enclosing region/loop block arguments (the canonical example is
// an `hc.for_range` induction variable typed `!hc.idx<"$join0">`) and
// bind any bare-sym `!hc.idx` we find. Without this the composed
// offset would leave such names as free symbols, and the launch-body
// lowering would have to resolve them ambiently — fragile, because
// the structured-loop converter rewrites the for_range to `scf.for`
// before the inner apply gets lowered, and at that point the original
// `!hc.idx<sym>` type is gone from the IR. Explicit operand binding
// here keeps the apply's free-sym set bounded to the launch geometry
// and kernel-arg shape syms.
static void bindAncestorBareSymbolBlockArgs(Operation *op, sym::Store &store,
                                            llvm::StringMap<Value> &bindings) {
  for (Block *block = op->getBlock(); block;) {
    for (BlockArgument arg : block->getArguments()) {
      StringRef name = pinsBareIdxSymbol(arg.getType(), store);
      if (name.empty())
        continue;
      bindings.try_emplace(name, arg);
    }
    Operation *parent = block->getParentOp();
    if (!parent)
      break;
    block = parent->getBlock();
  }
}

static FailureOr<Value>
composeAccessBaseOffset(ConversionPatternRewriter &rewriter, Operation *op,
                        Value preFlattenOperand, ValueRange shapedAux,
                        OperandRange indices) {
  FailureOr<ShapedAccessOperandInfo> info =
      validateShapedAccessOperand(preFlattenOperand);
  if (failed(info))
    return failure();

  MLIRContext *ctx = op->getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  FailureOr<SmallVector<ExprAttr>> indexExprs =
      buildAccessIndexExprs(ctx, store, indices);
  if (failed(indexExprs))
    return failure();
  auto offsetExpr =
      composeAccessOffsetExpr(ctx, info->layout, info->shape, *indexExprs);
  if (failed(offsetExpr))
    return failure();

  llvm::StringMap<Value> bindings;
  if (failed(bindShapedAuxImplicitSyms(info->shaped, shapedAux, bindings)))
    return failure();
  bindBareSymbolIndexOperands(indices, store, bindings);
  bindAncestorBareSymbolBlockArgs(op, store, bindings);

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
    registerIdentityConversion();
    registerShapedConversion();
    registerTupleConversion();
    registerFunctionConversion();
    registerCastMaterializations();
  }

private:
  void registerIdentityConversion() {
    addConversion([](Type t, SmallVectorImpl<Type> &results) {
      results.push_back(t);
      return success();
    });
  }

  void registerShapedConversion() {
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
  }

  void registerTupleConversion() {
    addConversion(
        [this](TupleType tuple,
               SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          SmallVector<Type> elements;
          if (failed(convertTypes(tuple.getTypes(), elements)))
            return failure();
          results.push_back(TupleType::get(tuple.getContext(), elements));
          return success();
        });
  }

  void registerFunctionConversion() {
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
  }

  void registerCastMaterializations() {
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    });
    addTargetMaterialization([](OpBuilder &builder, TypeRange resultTypes,
                                ValueRange inputs,
                                Location loc) -> SmallVector<Value> {
      auto cast =
          UnrealizedConversionCastOp::create(builder, loc, resultTypes, inputs);
      return SmallVector<Value>(cast.getResults());
    });
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

// Already-flat single-index loads against a layout-less source need no
// offset folding — the generic retype handles the operand type.
// Layout-bearing 1D sources (the buffer_view-with-strided-slice
// residual that `composeBufferViewLayout` produces) still need
// composition, so this returns true (i.e. the access needs
// composition) when either the indexing isn't 1D or the source's
// pre-flatten type carries a layout.
static bool needsAccessOffsetComposition(unsigned indexCount, Type sourceType) {
  if (indexCount != 1)
    return true;
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(sourceType);
  return shaped && shaped.getSymbolicLayout();
}

// Convert `t` via `converter` and verify the result is non-empty.
// Both `convertType` failure and an empty result indicate the access
// can't be lowered to its expected post-flatten shape.
static FailureOr<SmallVector<Type>>
convertResultTypeOrFailure(const TypeConverter &converter, Type t) {
  SmallVector<Type> result;
  if (failed(converter.convertType(t, result)))
    return failure();
  if (result.empty())
    return failure();
  return result;
}

// Splice the new op's primary result with its post-flatten aux values
// (resolved from the result-type bindings) and rewrite `op` against
// the combined sequence using the 1-to-N replacement entry point.
template <typename OpT>
static LogicalResult
replaceLoadWithAuxValues(OpT op, Value newResult,
                         llvm::StringMap<Value> &bindings,
                         ConversionPatternRewriter &rewriter) {
  auto auxValues = resolveResultAuxValues(rewriter, op.getLoc(),
                                          op.getResult().getType(), bindings);
  if (failed(auxValues))
    return failure();
  SmallVector<Value> replacement = {newResult};
  llvm::append_range(replacement, *auxValues);
  SmallVector<ValueRange> replacements = {replacement};
  rewriter.replaceOpWithMultiple(op, replacements);
  return success();
}

struct ComposeLoadOffsets : public ComposeAccessOffsetBase<HCLoadOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCLoadOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsAccessOffsetComposition(op.getIndices().size(),
                                      op.getBuffer().getType()))
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

    FailureOr<SmallVector<Type>> convertedResults = convertResultTypeOrFailure(
        *getTypeConverter(), op.getResult().getType());
    if (failed(convertedResults))
      return failure();

    auto newLoad =
        HCLoadOp::create(rewriter, op.getLoc(), convertedResults->front(),
                         flatBuffer, ValueRange{*base}, adaptor.getShape()[0],
                         /*layout=*/LayoutAttr{});

    llvm::StringMap<Value> bindings;
    noteOperandBindings(op.getBuffer().getType(), adaptor.getBuffer(),
                        bindings);
    return replaceLoadWithAuxValues(op, newLoad.getResult(), bindings,
                                    rewriter);
  }
};

struct ComposeVLoadOffsets : public ComposeAccessOffsetBase<HCVLoadOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCVLoadOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCVLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsAccessOffsetComposition(op.getIndices().size(),
                                      op.getSource().getType()))
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

    FailureOr<SmallVector<Type>> convertedResults = convertResultTypeOrFailure(
        *getTypeConverter(), op.getResult().getType());
    if (failed(convertedResults))
      return failure();

    auto newVLoad =
        HCVLoadOp::create(rewriter, op.getLoc(), convertedResults->front(),
                          flatSource, ValueRange{*base}, adaptor.getShape()[0],
                          /*layout=*/LayoutAttr{});

    llvm::StringMap<Value> bindings;
    noteOperandBindings(op.getSource().getType(), adaptor.getSource(),
                        bindings);
    return replaceLoadWithAuxValues(op, newVLoad.getResult(), bindings,
                                    rewriter);
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
// Bundle of pre-checked inputs the buffer-view flattener carries
// across its sub-stages: the flat carrier values, the pre-flatten
// source's shape/layout view, and the result-type halves.
struct BufferViewFlattenInputs {
  Value flatSource;
  ValueRange sourceAux;
  SymbolicallyShapedTypeInterface preFlattenSrc;
  ShapeAttr preShape;
  SymbolicallyShapedTypeInterface flatSourceShaped;
  SymbolicallyShapedTypeInterface flatResultShaped;
  Type flatResultType;
  Type origResultType;
};

// Pre-flatten buffer-view shape parity gate: the flattener has nothing
// to do when the indexing is already 1D and the source is 1D, and the
// strided-slice branch can't fold subscripts when the rank doesn't
// match the shape (unless a layout is in play, in which case the
// identity branch handles it).
static bool bufferViewNeedsFlatten(unsigned indexCount, ShapeAttr preShape,
                                   bool hasLayout) {
  size_t dimCount = preShape.getDims().size();
  if (dimCount == indexCount && dimCount <= 1)
    return false;
  if (!hasLayout && dimCount != indexCount)
    return false;
  return true;
}

// Validate operand and result shapes for the buffer-view flattener.
// `convertedBufferOperand` is the 1-to-N expansion of `op.getBuffer()`
// from the conversion adaptor — passing `ValueRange` instead of the
// adaptor type sidesteps the non-public-alias on the op class.
static FailureOr<BufferViewFlattenInputs>
prepareBufferViewFlatten(HCBufferViewOp op, ValueRange convertedBufferOperand,
                         const TypeConverter &converter) {
  if (convertedBufferOperand.empty())
    return failure();
  Value flatSource = convertedBufferOperand.front();
  ValueRange sourceAux = convertedBufferOperand.drop_front();

  auto preFlattenSrc =
      dyn_cast<SymbolicallyShapedTypeInterface>(op.getBuffer().getType());
  if (!preFlattenSrc)
    return failure();
  ShapeAttr preShape = preFlattenSrc.getSymbolicShape();
  if (!preShape)
    return failure();
  if (!bufferViewNeedsFlatten(op.getIndices().size(), preShape,
                              preFlattenSrc.getSymbolicLayout() != nullptr))
    return failure();

  Type origResultType = op.getResult().getType();
  FailureOr<SmallVector<Type>> convertedResults =
      convertResultTypeOrFailure(converter, origResultType);
  if (failed(convertedResults))
    return failure();
  Type flatResultType = convertedResults->front();

  auto flatResultShaped =
      dyn_cast<SymbolicallyShapedTypeInterface>(flatResultType);
  auto flatSourceShaped =
      dyn_cast<SymbolicallyShapedTypeInterface>(flatSource.getType());
  if (!flatResultShaped || !flatSourceShaped)
    return failure();

  return BufferViewFlattenInputs{
      flatSource,       sourceAux,        preFlattenSrc,  preShape,
      flatSourceShaped, flatResultShaped, flatResultType, origResultType};
}

// Replace `op` with the computed `flatValue` plus the per-result aux
// values resolved from the source operand's expansion. Mirrors the
// existing `replaceLoadWithAuxValues` pathway but typed for an
// `HCBufferViewOp`.
static LogicalResult
pushBufferViewReplacement(HCBufferViewOp op, ValueRange convertedBufferOperand,
                          Value flatValue, Type origResultType,
                          ConversionPatternRewriter &rewriter) {
  llvm::StringMap<Value> bindings;
  noteOperandBindings(op.getBuffer().getType(), convertedBufferOperand,
                      bindings);
  auto auxValues =
      resolveResultAuxValues(rewriter, op.getLoc(), origResultType, bindings);
  if (failed(auxValues))
    return failure();
  SmallVector<Value> replacement = {flatValue};
  llvm::append_range(replacement, *auxValues);
  SmallVector<ValueRange> replacements = {replacement};
  rewriter.replaceOpWithMultiple(op, replacements);
  return success();
}

// Identity case: the flat carriers match between source and result, so
// the view describes the same physical storage. Two distinct cases
// converge here. (1) Full-bind on a layout-less source where every
// scalar subscript hits an axis the surrounding flatten layout has
// already factored out; the source-side strides land on the same 1-D
// carrier the result wants. (2) Layout-bearing source where
// `inferBufferViewResult` substituted scalar-axis index values into
// the result's offset and dropped the corresponding shape syms —
// residual `storage_size` still names the same physical span, the
// flat carrier types match, and the only delta is the relabel of the
// per-axis aux set, which `resolveResultAuxValues` rebinds from the
// source's expansion. Cross-element-type view requests don't exist in
// the v0 surface; the carrier-type equality check rejects any flat-
// shape coincidence that would change the element type before it can
// silently miscompile.
static bool isBufferViewFlatIdentity(const BufferViewFlattenInputs &in) {
  return in.flatSourceShaped.getSymbolicShape() ==
             in.flatResultShaped.getSymbolicShape() &&
         in.flatSource.getType() == in.flatResultType;
}

// Locate the single slice axis among `indices`; reject anything
// non-idx, non-slice, or multiple slices.
static FailureOr<int64_t> findSingleSliceAxis(OperandRange indices) {
  int64_t sliceAxis = -1;
  for (auto [axis, idx] : llvm::enumerate(indices)) {
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
  return sliceAxis;
}

// Symbolic row-major stride at axis `k` of a static dim list: the
// product of dim sizes for every axis after `k`.
static FailureOr<sym::ExprHandle>
composeRowMajorStrideExpr(sym::Store &store, ArrayRef<Attribute> preDims,
                          size_t k) {
  auto oneE = sym::composeExprInt(store, 1);
  if (failed(oneE))
    return failure();
  sym::ExprHandle stride = *oneE;
  for (size_t j = k + 1; j < preDims.size(); ++j) {
    auto dim = dyn_cast<ExprAttr>(preDims[j]);
    if (!dim)
      return failure();
    auto next = sym::composeExprBinary(store, stride, sym::ExprBinaryOp::Mul,
                                       dim.getValue());
    if (failed(next))
      return failure();
    stride = *next;
  }
  return stride;
}

// Sum the per-scalar-axis `index * row_stride` contributions to a
// running flat base offset (identity when `indices` only carries the
// one slice axis).
static FailureOr<sym::ExprHandle>
composeScalarBaseOffsetExpr(sym::Store &store, OperandRange indices,
                            int64_t sliceAxis, ArrayRef<Attribute> preDims,
                            sym::ExprHandle zero) {
  sym::ExprHandle baseOffset = zero;
  for (auto [axis, idx] : llvm::enumerate(indices)) {
    if (axis == static_cast<size_t>(sliceAxis))
      continue;
    auto idxType = dyn_cast<IdxType>(idx.getType());
    if (!idxType || !idxType.getExpr())
      return failure();
    auto rowStride = composeRowMajorStrideExpr(store, preDims, axis);
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
  return baseOffset;
}

struct SliceTripleExprs {
  sym::ExprHandle lower;
  sym::ExprHandle upper;
  sym::ExprHandle step;
};

// Pull (lower, upper, step) from a slice producer's optional operands,
// substituting the Python slice defaults (`lower → 0`, `upper → axis
// size`, `step → 1`) whenever an operand is absent.
static FailureOr<SliceTripleExprs>
extractSliceTripleExprs(HCSliceExprOp sliceProducer, sym::ExprHandle zero,
                        sym::ExprHandle axisDim, sym::ExprHandle one) {
  auto exprFromOperand =
      [](Value v, sym::ExprHandle dflt) -> FailureOr<sym::ExprHandle> {
    if (!v)
      return dflt;
    auto t = dyn_cast<IdxType>(v.getType());
    if (!t || !t.getExpr())
      return failure();
    return t.getExpr().getValue();
  };
  auto lower = exprFromOperand(sliceProducer.getLower(), zero);
  if (failed(lower))
    return failure();
  auto upper = exprFromOperand(sliceProducer.getUpper(), axisDim);
  if (failed(upper))
    return failure();
  auto step = exprFromOperand(sliceProducer.getStep(), one);
  if (failed(step))
    return failure();
  return SliceTripleExprs{*lower, *upper, *step};
}

// Compose the flat (lower, upper, step) triple: each of the lower /
// upper bounds gets `slice_X * row_stride` summed onto the scalar-axis
// base offset; step is just `slice.step * row_stride` (no base added).
static FailureOr<SliceTripleExprs>
composeFlatSliceTripleExprs(sym::Store &store, sym::ExprHandle baseOffset,
                            sym::ExprHandle sliceRowStride,
                            const SliceTripleExprs &slice) {
  auto mul = [&](sym::ExprHandle a, sym::ExprHandle b) {
    return sym::composeExprBinary(store, a, sym::ExprBinaryOp::Mul, b);
  };
  auto add = [&](sym::ExprHandle a, sym::ExprHandle b) {
    return sym::composeExprBinary(store, a, sym::ExprBinaryOp::Add, b);
  };

  auto lowerContrib = mul(slice.lower, sliceRowStride);
  if (failed(lowerContrib))
    return failure();
  auto flatLower = add(baseOffset, *lowerContrib);
  if (failed(flatLower))
    return failure();
  auto upperContrib = mul(slice.upper, sliceRowStride);
  if (failed(upperContrib))
    return failure();
  auto flatUpper = add(baseOffset, *upperContrib);
  if (failed(flatUpper))
    return failure();
  auto flatStep = mul(slice.step, sliceRowStride);
  if (failed(flatStep))
    return failure();
  return SliceTripleExprs{*flatLower, *flatUpper, *flatStep};
}

// Build the symbol-binding map the same way `composeAccessBaseOffset`
// does for the per-access patterns: the source operand's 1-to-N
// expansion supplies dim / stride aux; idx-typed subscripts bind
// their own bare symbol; ancestor block args (loop induction vars)
// bind any bare-sym `!hc.idx` they carry. `materializeOffsetSSA`
// emits `hc.idx_apply` ops with the right operand list so the
// launch-body lowering downstream picks them up by SSA, not by
// ambient resolution.
static llvm::StringMap<Value>
collectBufferViewBindings(HCBufferViewOp op, const BufferViewFlattenInputs &in,
                          sym::Store &store) {
  llvm::StringMap<Value> bindings;
  SmallVector<std::string> implicitSyms = collectImplicitSyms(in.preFlattenSrc);
  if (in.sourceAux.size() == implicitSyms.size())
    for (auto [name, value] : llvm::zip_equal(implicitSyms, in.sourceAux))
      bindings[name] = value;
  bindBareSymbolIndexOperands(op.getIndices(), store, bindings);
  bindAncestorBareSymbolBlockArgs(op.getOperation(), store, bindings);
  return bindings;
}

// Materialize the flat slice triple as SSA, build the new
// `hc.slice_expr` + `hc.buffer_view`, and return the new view value.
static Value emitFlatBufferViewSlice(HCBufferViewOp op, Type flatResultType,
                                     Value flatSource,
                                     const SliceTripleExprs &flatTriple,
                                     const llvm::StringMap<Value> &bindings,
                                     ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = op.getContext();
  auto buildIdx = [&](sym::ExprHandle e) -> Value {
    return materializeOffsetSSA(rewriter, op.getLoc(), ExprAttr::get(ctx, e),
                                bindings);
  };
  Value newLower = buildIdx(flatTriple.lower);
  Value newUpper = buildIdx(flatTriple.upper);
  Value newStep = buildIdx(flatTriple.step);
  auto sliceType = SliceType::get(ctx, newLower.getType(), newUpper.getType(),
                                  newStep.getType());
  Value newSlice = HCSliceExprOp::create(rewriter, op.getLoc(), sliceType,
                                         newLower, newUpper, newStep)
                       .getResult();
  return HCBufferViewOp::create(rewriter, op.getLoc(), flatResultType,
                                flatSource, ValueRange{newSlice})
      .getResult();
}

struct SymZeroOne {
  sym::ExprHandle zero;
  sym::ExprHandle one;
};

// Compose the (0, 1) symbolic constants together so the caller pays
// for the failure short-circuit once instead of twice.
static FailureOr<SymZeroOne> composeZeroOneExprs(sym::Store &store) {
  auto z = sym::composeExprInt(store, 0);
  auto o = sym::composeExprInt(store, 1);
  if (failed(z) || failed(o))
    return failure();
  return SymZeroOne{*z, *o};
}

// Synthesize the strided-slice flat view for an `hc.buffer_view` that
// the identity branch couldn't handle: exactly one slice subscript
// (the rest must be scalar-typed `!hc.idx`), composed against the
// pre-flatten source's row-major stride layout.
static FailureOr<Value>
synthesizeFlatStridedSliceView(HCBufferViewOp op,
                               const BufferViewFlattenInputs &in,
                               ConversionPatternRewriter &rewriter) {
  if (in.preFlattenSrc.getSymbolicLayout())
    return failure();
  FailureOr<int64_t> sliceAxis = findSingleSliceAxis(op.getIndices());
  if (failed(sliceAxis))
    return failure();

  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  ArrayRef<Attribute> preDims = in.preShape.getDims();

  FailureOr<SymZeroOne> zeroOne = composeZeroOneExprs(store);
  if (failed(zeroOne))
    return failure();

  FailureOr<sym::ExprHandle> baseOffset = composeScalarBaseOffsetExpr(
      store, op.getIndices(), *sliceAxis, preDims, zeroOne->zero);
  if (failed(baseOffset))
    return failure();

  auto sliceProducer =
      op.getIndices()[*sliceAxis].getDefiningOp<HCSliceExprOp>();
  if (!sliceProducer)
    return failure();
  auto sliceAxisDim = dyn_cast<ExprAttr>(preDims[*sliceAxis]);
  if (!sliceAxisDim)
    return failure();
  FailureOr<SliceTripleExprs> sliceTriple = extractSliceTripleExprs(
      sliceProducer, zeroOne->zero, sliceAxisDim.getValue(), zeroOne->one);
  if (failed(sliceTriple))
    return failure();
  auto sliceRowStride = composeRowMajorStrideExpr(store, preDims, *sliceAxis);
  if (failed(sliceRowStride))
    return failure();

  FailureOr<SliceTripleExprs> flatTriple = composeFlatSliceTripleExprs(
      store, *baseOffset, *sliceRowStride, *sliceTriple);
  if (failed(flatTriple))
    return failure();

  llvm::StringMap<Value> bindings = collectBufferViewBindings(op, in, store);
  return emitFlatBufferViewSlice(op, in.flatResultType, in.flatSource,
                                 *flatTriple, bindings, rewriter);
}

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
    ValueRange bufferOperand = adaptor.getBuffer();
    FailureOr<BufferViewFlattenInputs> in =
        prepareBufferViewFlatten(op, bufferOperand, *getTypeConverter());
    if (failed(in))
      return failure();

    if (isBufferViewFlatIdentity(*in))
      return pushBufferViewReplacement(op, bufferOperand, in->flatSource,
                                       in->origResultType, rewriter);

    FailureOr<Value> newView =
        synthesizeFlatStridedSliceView(op, *in, rewriter);
    if (failed(newView))
      return failure();
    return pushBufferViewReplacement(op, bufferOperand, *newView,
                                     in->origResultType, rewriter);
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
// Compose one per-operand offset array against its operand's
// pre-flatten layout / shape. Operands that aren't shaped (or are
// already-flat) pass through unchanged. Returns failure to bail the
// whole rewrite to `RetypeAnyHCOp`. `composed` flips to true when the
// folded result actually differs from the input — a rank-1 layout-less
// operand round-trips through the identity layout to itself, and we
// don't want to claim a rewrite happened when nothing on the surface
// moved.
//
// Layout-less operands (including buffers that haven't picked up the
// default strided layout — `hc-canonicalize-layouts` only attaches one
// for kernel-arg buffers) fall back to the identity layout, the
// canonical contract for layout-free shaped types. Same path
// `composeAccessOffsetExpr` takes for tensor / vector operands.
static FailureOr<ArrayAttr> composeGenericOperandOffsets(MLIRContext *ctx,
                                                         Value origOperand,
                                                         ArrayAttr perAxis,
                                                         bool &composed) {
  auto shaped =
      dyn_cast<SymbolicallyShapedTypeInterface>(origOperand.getType());
  if (!shaped)
    return perAxis;
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return perAxis;
  LayoutAttr layout = shaped.getSymbolicLayout();
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
  if (result != perAxis)
    composed = true;
  return result;
}

// Apply `composeGenericOperandOffsets` across an operand range,
// collecting the per-operand results (or short-circuiting to failure).
static FailureOr<SmallVector<Attribute>>
composeGenericOffsetsArray(MLIRContext *ctx, OperandRange operands,
                           ArrayAttr offsets, bool &composed) {
  SmallVector<Attribute> result;
  result.reserve(offsets.size());
  for (auto [operand, perAxis] :
       llvm::zip_equal(operands, offsets.getAsRange<ArrayAttr>())) {
    auto out = composeGenericOperandOffsets(ctx, operand, perAxis, composed);
    if (failed(out))
      return failure();
    result.push_back(*out);
  }
  return result;
}

// Pick the first entry out of every ValueRange in `expansion` (the
// flat carrier; trailing aux belongs to the operand's own expansion).
// Returns failure when any range is empty or, if `requireSingleton` is
// set, when any range carries more than one value (matches the
// `iter_bounds` contract that's strictly 1:1).
static FailureOr<SmallVector<Value>>
collectFlatLeadingValues(ArrayRef<ValueRange> expansion,
                         bool requireSingleton = false) {
  SmallVector<Value> result;
  result.reserve(expansion.size());
  for (ValueRange range : expansion) {
    if (range.empty())
      return failure();
    if (requireSingleton && range.size() != 1)
      return failure();
    result.push_back(range.front());
  }
  return result;
}

// Collect free symbols referenced by composed offset arrays, minus any
// iter syms (those stay scoped to the `hc.generic` body and are
// substituted per-lane by `hc-lower-generic`).
static void collectAmbientSymsFromOffsets(ArrayRef<Attribute> perOperandAttrs,
                                          const llvm::StringSet<> &iterSymSet,
                                          llvm::StringSet<> &ambientNeeded) {
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
}

// Body `hc.idx_apply` / `hc.pred_apply` ops can reference ambient syms
// directly (post-flatten, the body authoring convention is that any
// sym not on the apply's `symbols` list is either an iter sym scoped
// to the body or an ambient sym the surrounding `hc.generic` is
// responsible for plumbing). Pick those up too so `hc-lower-generic`
// can seed the per-lane scope from `ambient_idxs` and the second
// `hc-lower-launch-body` invocation finds bindings for them via
// `seedAmbientScope` inside the unrolled body. The `hc.load_mask`
// rewrite in `hc-load-store-to-generic` is today the only emitter that
// puts ambient-referencing applies inside the body (its predicate is
// `(lo + step*i_k) < D_k` with `lo` / `D_k` being kernel-arg or
// launch-geometry syms) — extending the walk now keeps the contract
// general.
// Helper: visit each unbound free symbol name surfaced by `walker`,
// skipping those already named on the apply op and those that are
// part of the iter-sym set (which are bound by the surrounding
// `hc.generic`).
static void noteFreeAmbientSyms(
    ArrayAttr alreadyBound, const llvm::StringSet<> &iterSymSet,
    llvm::StringSet<> &ambientNeeded,
    llvm::function_ref<void(llvm::function_ref<void(StringRef)>)> walker) {
  llvm::StringSet<> already;
  for (Attribute n : alreadyBound)
    already.insert(cast<StringAttr>(n).getValue());
  walker([&](StringRef name) {
    if (already.contains(name) || iterSymSet.contains(name))
      return;
    ambientNeeded.insert(name);
  });
}

static void collectAmbientSymsFromIdxApply(HCIdxApplyOp idx,
                                           const llvm::StringSet<> &iterSymSet,
                                           llvm::StringSet<> &ambientNeeded) {
  auto idxTy = dyn_cast<IdxType>(idx.getResult().getType());
  if (!idxTy || !idxTy.getExpr())
    return;
  noteFreeAmbientSyms(idx.getSymbolsAttr(), iterSymSet, ambientNeeded,
                      [&](llvm::function_ref<void(StringRef)> cb) {
                        sym::walkSymbolNames(idxTy.getExpr().getValue(), cb);
                      });
}

static void collectAmbientSymsFromPredApply(HCPredApplyOp predOp,
                                            const llvm::StringSet<> &iterSymSet,
                                            llvm::StringSet<> &ambientNeeded) {
  auto predTy = dyn_cast<PredType>(predOp.getResult().getType());
  if (!predTy || !predTy.getPred())
    return;
  noteFreeAmbientSyms(predOp.getSymbolsAttr(), iterSymSet, ambientNeeded,
                      [&](llvm::function_ref<void(StringRef)> cb) {
                        sym::walkSymbolNames(predTy.getPred().getValue(), cb);
                      });
}

static void
collectAmbientSymsFromBodyApplies(Block &body,
                                  const llvm::StringSet<> &iterSymSet,
                                  llvm::StringSet<> &ambientNeeded) {
  for (Operation &nested : body) {
    if (auto idx = dyn_cast<HCIdxApplyOp>(&nested))
      collectAmbientSymsFromIdxApply(idx, iterSymSet, ambientNeeded);
    else if (auto predOp = dyn_cast<HCPredApplyOp>(&nested))
      collectAmbientSymsFromPredApply(predOp, iterSymSet, ambientNeeded);
  }
}

// Carry over ambient bindings the source op already had — the
// pre-flatten emitters may have left them empty, but if a prior pass
// populated them we don't want to drop the SSA edge silently.
static void
notePreExistingAmbientBindings(ArrayRef<ValueRange> ambientIdxsExpansion,
                               HCGenericOp op, llvm::StringMap<Value> &bindings,
                               llvm::StringSet<> &ambientNeeded) {
  for (auto [val, symAttr] :
       llvm::zip_equal(ambientIdxsExpansion,
                       op.getAmbientIdxSymsAttr().getAsRange<StringAttr>())) {
    if (val.empty())
      continue;
    bindings.try_emplace(symAttr.getValue(), val.front());
    ambientNeeded.insert(symAttr.getValue());
  }
}

// Lex-sort the ambient sym names for deterministic operand order, then
// pin bindings whose SSA we resolved. The rest stay free in the offset
// expression and `hc.idx_apply`'s severing form handles them via the
// downstream ambient-context resolution.
static void
buildAmbientOperandLists(const llvm::StringSet<> &ambientNeeded,
                         const llvm::StringMap<Value> &bindings,
                         OpBuilder &rewriter,
                         SmallVectorImpl<Value> &ambientIdxsVec,
                         SmallVectorImpl<Attribute> &ambientSymsVec) {
  SmallVector<StringRef> ambientNames(ambientNeeded.keys().begin(),
                                      ambientNeeded.keys().end());
  llvm::sort(ambientNames);
  for (StringRef name : ambientNames) {
    auto it = bindings.find(name);
    if (it == bindings.end())
      continue;
    ambientIdxsVec.push_back(it->second);
    ambientSymsVec.push_back(rewriter.getStringAttr(name));
  }
}

// Convert each result type via the 1-to-N converter and split into
// (flat leading types + per-result widths). Empty conversions or
// outright failures bail to the catch-all retyper.
static LogicalResult
convertGenericResultTypesToFlat(HCGenericOp op, const TypeConverter &converter,
                                SmallVectorImpl<Type> &flatResultTypes,
                                SmallVectorImpl<unsigned> &resultWidths) {
  flatResultTypes.reserve(op.getNumResults());
  resultWidths.reserve(op.getNumResults());
  for (Type resultType : op.getResultTypes()) {
    SmallVector<Type> converted;
    if (failed(converter.convertType(resultType, converted)))
      return failure();
    if (converted.empty())
      return failure();
    flatResultTypes.push_back(converted.front());
    resultWidths.push_back(converted.size());
  }
  return success();
}

// True iff any result type or any operand type would actually change
// across the rewrite. Without this gate the driver loops on a no-op
// rewrite.
static bool
genericTypesChangedAcrossRewrite(HCGenericOp op, ArrayRef<Type> flatResultTypes,
                                 ArrayRef<ValueRange> operandExpansion) {
  for (auto [oldT, newT] :
       llvm::zip_equal(op.getResultTypes(), flatResultTypes))
    if (oldT != newT)
      return true;
  for (auto [oldVal, range] : llvm::zip(op.getOperands(), operandExpansion))
    if (range.size() != 1 || oldVal.getType() != range.front().getType())
      return true;
  return false;
}

// Carry over any extra discardable attributes (e.g. location-name
// hints) the original op picked up before we got here. The named
// attributes the builder wrote — iter_syms, iter_kinds, ins/outs
// offsets, segment sizes — already match.
static void copyDiscardableHCGenericAttrs(HCGenericOp op, HCGenericOp newOp) {
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
}

// Pair each new flat result with the aux values its pre-flatten type
// expects; when bindings already have an SSA edge for an aux name use
// it instead of materialising a fresh ambient apply.
static LogicalResult
replaceGenericWithAuxValues(HCGenericOp op, HCGenericOp newOp,
                            llvm::StringMap<Value> &bindings,
                            ConversionPatternRewriter &rewriter) {
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

// Bundles every per-operand precomputation a `ComposeGenericOffsets`
// rewrite needs: composed offset arrays, the flat carrier slice of
// each variadic operand, and the bool that flags whether the offset
// composition step changed anything (used as a fixed-point guard).
struct ComposedGenericOperands {
  SmallVector<Attribute> insOffsets;
  SmallVector<Attribute> outsOffsets;
  SmallVector<Value> iterBounds;
  SmallVector<Value> flatIns;
  SmallVector<Value> flatOuts;
  bool composed;
};

using HCGenericOpOneToNAdaptor =
    typename OpConversionPattern<HCGenericOp>::OneToNOpAdaptor;

static FailureOr<ComposedGenericOperands>
composeAndSliceGenericOperands(HCGenericOp op, HCGenericOpOneToNAdaptor adaptor,
                               MLIRContext *ctx) {
  ComposedGenericOperands out;
  out.composed = false;
  FailureOr<SmallVector<Attribute>> insOff = composeGenericOffsetsArray(
      ctx, op.getIns(), op.getInsOffsetsAttr(), out.composed);
  if (failed(insOff))
    return failure();
  FailureOr<SmallVector<Attribute>> outsOff = composeGenericOffsetsArray(
      ctx, op.getOuts(), op.getOutsOffsetsAttr(), out.composed);
  if (failed(outsOff))
    return failure();
  if (adaptor.getIterBounds().size() != op.getIterBounds().size())
    return failure();
  // Flat carrier slicing matches RetypeAnyHCOp's contract — each
  // 1-to-N adapter range hands back the flat shaped value as the
  // leading entry; trailing aux values feed the result-expansion
  // binding map.
  FailureOr<SmallVector<Value>> iterBounds = collectFlatLeadingValues(
      adaptor.getIterBounds(), /*requireSingleton=*/true);
  if (failed(iterBounds))
    return failure();
  FailureOr<SmallVector<Value>> flatIns =
      collectFlatLeadingValues(adaptor.getIns());
  if (failed(flatIns))
    return failure();
  FailureOr<SmallVector<Value>> flatOuts =
      collectFlatLeadingValues(adaptor.getOuts());
  if (failed(flatOuts))
    return failure();
  out.insOffsets = std::move(*insOff);
  out.outsOffsets = std::move(*outsOff);
  out.iterBounds = std::move(*iterBounds);
  out.flatIns = std::move(*flatIns);
  out.flatOuts = std::move(*flatOuts);
  return out;
}

// Build the sym → SSA map and the ambient operand list a generic
// rewrite ships to the new op. Captures: per-operand expansions,
// ancestor block-arg bindings, ambient symbols transitively needed by
// composed offsets and body applies, and any pre-existing ambient
// bindings on the source op.
static void buildGenericBindingsAndAmbient(
    HCGenericOp op, HCGenericOpOneToNAdaptor adaptor,
    ArrayRef<Attribute> insOffsets, ArrayRef<Attribute> outsOffsets,
    ConversionPatternRewriter &rewriter, llvm::StringMap<Value> &bindings,
    SmallVectorImpl<Value> &ambientIdxsVec,
    SmallVectorImpl<Attribute> &ambientSymsVec) {
  // Pull the shape-preserving sym-name bindings off every operand
  // expansion so result aux can be sourced from a matching name
  // before falling back to an ambient `hc.idx_apply`.
  for (auto [orig, range] : llvm::zip_equal(op.getIns(), adaptor.getIns()))
    noteOperandBindings(orig.getType(), range, bindings);
  for (auto [orig, range] : llvm::zip_equal(op.getOuts(), adaptor.getOuts()))
    noteOperandBindings(orig.getType(), range, bindings);

  // Capture ambient sym → SSA bindings now, while the kernel-arg
  // bundle UCC chain, gpu.launch block args, and structured-loop
  // induction vars are all still HC-typed and reachable. Walking
  // ancestor blocks here also picks up `$joinN` from `hc.for_range`'s
  // IV — `hc-lower-launch-body` will later rewrite for_range to
  // scf.for and strip the `!hc.idx<sym>` payload off the IV, but by
  // then `ambient_idxs` already holds the SSA edge and the
  // launch-body type converter only changes the operand's type (the
  // sym name lives on `ambient_idx_syms`).
  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  collectAncestorIdxBindings(store, op, bindings);

  llvm::StringSet<> iterSymSet;
  for (Attribute s : op.getIterSymsAttr())
    iterSymSet.insert(cast<StringAttr>(s).getValue());
  llvm::StringSet<> ambientNeeded;
  collectAmbientSymsFromOffsets(insOffsets, iterSymSet, ambientNeeded);
  collectAmbientSymsFromOffsets(outsOffsets, iterSymSet, ambientNeeded);
  if (!op.getBody().empty())
    collectAmbientSymsFromBodyApplies(op.getBody().front(), iterSymSet,
                                      ambientNeeded);
  notePreExistingAmbientBindings(adaptor.getAmbientIdxs(), op, bindings,
                                 ambientNeeded);
  buildAmbientOperandLists(ambientNeeded, bindings, rewriter, ambientIdxsVec,
                           ambientSymsVec);
}

struct ComposeGenericOffsets : public ComposeAccessOffsetBase<HCGenericOp> {
  using ComposeAccessOffsetBase::ComposeAccessOffsetBase;
  using Base = OpConversionPattern<HCGenericOp>;
  using OneToNOpAdaptor = typename Base::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(HCGenericOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    FailureOr<ComposedGenericOperands> prepared =
        composeAndSliceGenericOperands(op, adaptor, ctx);
    if (failed(prepared))
      return failure();

    llvm::StringMap<Value> bindings;
    SmallVector<Value> ambientIdxsVec;
    SmallVector<Attribute> ambientSymsVec;
    buildGenericBindingsAndAmbient(op, adaptor, prepared->insOffsets,
                                   prepared->outsOffsets, rewriter, bindings,
                                   ambientIdxsVec, ambientSymsVec);

    SmallVector<Type> flatResultTypes;
    SmallVector<unsigned> resultWidths;
    if (failed(convertGenericResultTypesToFlat(op, *getTypeConverter(),
                                               flatResultTypes, resultWidths)))
      return failure();

    if (!prepared->composed && !genericTypesChangedAcrossRewrite(
                                   op, flatResultTypes, adaptor.getOperands()))
      return failure();

    auto newOp = HCGenericOp::create(
        rewriter, op.getLoc(), flatResultTypes, op.getIterSymsAttr(),
        ValueRange(prepared->iterBounds), op.getIterKindsAttr(),
        ValueRange(prepared->flatIns), ValueRange(prepared->flatOuts),
        /*ambient_idxs=*/ValueRange(ambientIdxsVec),
        rewriter.getArrayAttr(ambientSymsVec),
        ArrayAttr::get(ctx, prepared->insOffsets),
        ArrayAttr::get(ctx, prepared->outsOffsets));
    copyDiscardableHCGenericAttrs(op, newOp);

    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(
            rewriter.convertRegionTypes(&newOp.getBody(), *getTypeConverter())))
      return failure();

    return replaceGenericWithAuxValues(op, newOp, bindings, rewriter);
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

    SmallVector<Type> convertedResultTypes;
    SmallVector<unsigned> resultWidths;
    if (failed(convertResultTypes(op->getResultTypes(), *getTypeConverter(),
                                  convertedResultTypes, resultWidths)))
      return failure();

    if (!retypeWouldChangeOp(op, operands, convertedResultTypes))
      return failure();

    SmallVector<Type> flatResultTypes;
    if (failed(narrowResultTypesToFlat(convertedResultTypes, resultWidths,
                                       op->getNumResults(), flatResultTypes)))
      return failure();

    Operation *newOp = cloneOpWithFlatTypes(op, operands, flatResultTypes,
                                            *getTypeConverter(), rewriter);
    if (!newOp)
      return failure();

    return replaceWithRetypedAuxValues(op, newOp, operands, rewriter);
  }

  // Cheap no-op check: skip if every result and every operand is
  // already legal under the converter (the driver would re-fire us
  // forever otherwise).
  static bool retypeWouldChangeOp(Operation *op, ArrayRef<ValueRange> operands,
                                  ArrayRef<Type> convertedResultTypes) {
    if (op->getNumResults() != convertedResultTypes.size())
      return true;
    for (auto [oldT, newT] :
         llvm::zip_equal(op->getResultTypes(), convertedResultTypes))
      if (oldT != newT)
        return true;
    for (auto [oldVal, range] : llvm::zip_equal(op->getOperands(), operands))
      if (range.size() != 1 || oldVal.getType() != range.front().getType())
        return true;
    return false;
  }

  // Pick the leading (flat) result type out of each 1-to-N expansion;
  // the new op only carries those (the trailing aux idx values are
  // SSA-generated alongside, not on the op surface). A zero-width
  // expansion means a result the converter can't produce a flat type
  // for, which the caller must treat as a hard failure.
  static LogicalResult
  narrowResultTypesToFlat(ArrayRef<Type> convertedResultTypes,
                          ArrayRef<unsigned> resultWidths, unsigned numResults,
                          SmallVectorImpl<Type> &flatResultTypes) {
    flatResultTypes.reserve(numResults);
    unsigned offset = 0;
    for (unsigned width : resultWidths) {
      if (width == 0)
        return failure();
      flatResultTypes.push_back(convertedResultTypes[offset]);
      offset += width;
    }
    return success();
  }

  // Build the replacement op with flat-only operands and result types.
  // Inlines the original op's regions verbatim, then re-runs the
  // converter on each region body — `inlineRegionBefore` doesn't
  // touch block-arg types, and `hc.for_range`'s body iter-arg block
  // args must match the converted `iter_inits` or the parent
  // verifier rejects the op. For ops whose body block args don't
  // get flattened this is a no-op.
  static Operation *cloneOpWithFlatTypes(Operation *op,
                                         ArrayRef<ValueRange> operands,
                                         ArrayRef<Type> flatResultTypes,
                                         const TypeConverter &converter,
                                         ConversionPatternRewriter &rewriter) {
    SmallVector<Value> flatOperands = flatOperandsOnly(operands);
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
    for (Region &region : newOp->getRegions()) {
      if (region.empty())
        continue;
      if (failed(rewriter.convertRegionTypes(&region, converter)))
        return nullptr;
    }
    return newOp;
  }

  // Pair each new flat result with the aux values its pre-flatten
  // type expects, sourcing each aux name from the operand expansion
  // bindings when possible (so a result sym matches an operand sym
  // instead of a fresh ambient apply).
  static LogicalResult
  replaceWithRetypedAuxValues(Operation *op, Operation *newOp,
                              ArrayRef<ValueRange> operands,
                              ConversionPatternRewriter &rewriter) {
    llvm::StringMap<Value> bindings;
    for (auto [origOperand, range] :
         llvm::zip_equal(op->getOperands(), operands))
      noteOperandBindings(origOperand.getType(), range, bindings);

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
// Resolve the (kind, axis) classification for an implicit sym tied to
// an aux slot: stride syms are recognized from the canonical
// `$Sx<axis>` spelling; dim syms match a position in the buffer's
// shape axis names. A dim that doesn't match any shape axis leaks in
// via a non-shape source (`storage_size`, layout params); the host
// wrapper can't bind it from `hc_get_dim` alone, so we drop it.
struct FlattenAuxKindAxis {
  StringRef kind;
  int64_t axis;
};
static std::optional<FlattenAuxKindAxis>
classifyImplicitAuxSym(StringRef name, ArrayRef<StringRef> axisSyms) {
  if (std::optional<unsigned> strideAxis = parseStrideAxis(name))
    return FlattenAuxKindAxis{"stride", static_cast<int64_t>(*strideAxis)};
  for (auto [a, axisSym] : llvm::enumerate(axisSyms))
    if (axisSym == name)
      return FlattenAuxKindAxis{"dim", static_cast<int64_t>(a)};
  return std::nullopt;
}

// Build one `(aux_of, axis, kind)` dictionary entry for the given aux
// slot, keyed by its post-flatten function-arg index.
static NamedAttribute
buildFlattenAuxArgEntry(MLIRContext *ctx, IntegerType i64Type,
                        unsigned auxPostIdx, unsigned flatCarrierIdx,
                        const FlattenAuxKindAxis &kindAxis) {
  SmallVector<NamedAttribute> auxEntries;
  auxEntries.emplace_back(StringAttr::get(ctx, "aux_of"),
                          IntegerAttr::get(i64Type, flatCarrierIdx));
  auxEntries.emplace_back(StringAttr::get(ctx, "axis"),
                          IntegerAttr::get(i64Type, kindAxis.axis));
  auxEntries.emplace_back(StringAttr::get(ctx, "kind"),
                          StringAttr::get(ctx, kindAxis.kind));

  SmallString<8> key;
  Twine(auxPostIdx).toVector(key);
  return NamedAttribute(StringAttr::get(ctx, key),
                        DictionaryAttr::get(ctx, auxEntries));
}

// Walk the implicit-sym list for one buffer arg and append one
// `(aux_of, axis, kind)` entry per recognized (dim or stride) aux
// slot. Aux slots whose sym leaks in via a non-shape source are
// silently dropped (see `classifyImplicitAuxSym`).
static void
noteFlattenAuxArgsForBuffer(MLIRContext *ctx, IntegerType i64Type,
                            unsigned flatCarrierIdx,
                            SymbolicallyShapedTypeInterface shaped,
                            SmallVectorImpl<NamedAttribute> &entries) {
  SmallVector<StringRef> axisSyms;
  if (ShapeAttr shape = shaped.getSymbolicShape())
    for (Attribute dim : shape.getDims())
      axisSyms.push_back(bareDimSymbolName(dim));

  SmallVector<std::string> implicitSyms = collectImplicitSyms(shaped);
  for (auto [i, sym] : llvm::enumerate(implicitSyms)) {
    unsigned auxPostIdx = flatCarrierIdx + 1 + i;
    std::optional<FlattenAuxKindAxis> kindAxis =
        classifyImplicitAuxSym(StringRef(sym), axisSyms);
    if (!kindAxis)
      continue;
    entries.push_back(buildFlattenAuxArgEntry(ctx, i64Type, auxPostIdx,
                                              flatCarrierIdx, *kindAxis));
  }
}

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
    noteFlattenAuxArgsForBuffer(ctx, i64Type, newIdx, shaped, entries);
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

// Determine the rank of a single `hc.generic` operand for the
// post-flatten parity check. `nullopt` means the operand isn't
// rank-constrained at this boundary and shouldn't gate the legality
// decision (e.g. `hc.undef`-typed sentinels or non-shaped scalars).
static std::optional<size_t> hcGenericOperandRank(Type t) {
  if (isHCUndefType(t))
    return std::nullopt;
  if (isa<PtrType>(t))
    return size_t{1};
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t))
    if (ShapeAttr shape = shaped.getSymbolicShape())
      return shape.getDims().size();
  return std::nullopt;
}

// `hc.generic` legality gates on offset-array rank parity: a type-only
// retype with no offset composition leaves nD arrays sitting on
// now-1D operands, which the post-flatten verifier rejects. This
// returns false (i.e. the op is illegal) the moment any operand's
// offset count diverges from its rank, forcing the driver back
// through `ComposeGenericOffsets`.
static bool checkHCGenericRoleParity(OperandRange ops, ArrayAttr offsets) {
  for (auto [val, off] :
       llvm::zip_equal(ops, offsets.getAsRange<ArrayAttr>())) {
    std::optional<size_t> rank = hcGenericOperandRank(val.getType());
    if (rank && off.size() != *rank)
      return false;
  }
  return true;
}

static bool isHCGenericLegalAtFlattenBoundary(HCGenericOp generic,
                                              const TypeConverter &converter) {
  if (!converter.isLegal(generic.getOperation()))
    return false;
  if (!checkHCGenericRoleParity(generic.getIns(), generic.getInsOffsetsAttr()))
    return false;
  if (!checkHCGenericRoleParity(generic.getOuts(),
                                generic.getOutsOffsetsAttr()))
    return false;
  return true;
}

// HC's symbol-carrying ops carry their signature in a `function_type`
// attribute; `converter.isLegal(op)` only inspects operand/result
// types, which are zero on these ops. Match the upstream
// `FunctionOpInterface` legality rule explicitly. Returns nullopt
// when the op isn't a symbol-carrier (caller should fall through to
// the generic legality rules).
static std::optional<bool>
hcSymbolSignatureLegality(Operation *op, const TypeConverter &converter) {
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
  return std::nullopt;
}

// Top-level legality predicate for `markUnknownOpDynamicallyLegal`.
// Combines the upstream `FunctionOpInterface` rule, the func/return
// op rule, the HC symbol-signature rule, the HCGenericOp parity rule,
// and the HC dialect operand/result rule. Anything else is legal.
static bool isFlattenLegalAtPassBoundary(Operation *op,
                                         const TypeConverter &converter,
                                         MLIRContext *ctx) {
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
  if (std::optional<bool> sigLegal = hcSymbolSignatureLegality(op, converter))
    return *sigLegal;
  if (auto generic = dyn_cast<HCGenericOp>(op))
    return isHCGenericLegalAtFlattenBoundary(generic, converter);
  // HC dialect ops are legal iff every operand and every result type
  // is already in its converted form. The `RetypeAnyHCOp` pattern
  // takes care of the rebuild when one side still carries the
  // pre-flatten shape.
  if (op->getDialect() == ctx->getLoadedDialect<HCDialect>())
    return converter.isLegal(op);
  return true;
}

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
      return isFlattenLegalAtPassBoundary(op, converter, ctx);
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
