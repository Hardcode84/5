// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-flatten-with-layouts`: every shaped value drops its `#hc.layout`
// and collapses its shape to one entry. Tensors/vectors get a
// `storage_size_expr` (from layout or dim product); buffers collapse
// to `[?]` (host-owned allocation).
//
// 1-to-N converter: each shaped operand expands to `(flat_value,
// idx_aux_0, ..., idx_aux_{k-1})` -- one `!hc.idx<sym>` per implicit
// free symbol (shape syms + layout's free syms minus `index_syms`).
// Names sorted lex for determinism.
//
// Access ops (`hc.load`/`hc.vload`/`hc.store`) and `hc.generic`
// compose their per-axis offset arrays into a single 1D offset
// during conversion (layout still readable on pre-conversion type).
// `hc.buffer_view` / `hc.vec` keep their shape. `hc.as_layout`
// drops unconditionally.

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

// Implicit free syms of a shaped type: names in dim exprs + layout's
// offset/storage_size/params, minus `index_syms` and `shape_syms`.
// Post-flatten 1-to-N expansion's tail: one `!hc.idx<name>` per
// returned name, same order. Null `expr` is a no-op.
static void noteExprSymbolNames(ExprAttr expr, llvm::StringSet<> &seen) {
  if (!expr)
    return;
  sym::walkSymbolNames(expr.getValue(),
                       [&](StringRef name) { seen.insert(name); });
}

static void noteSymsInSymbolicShape(SymbolicallyShapedTypeInterface shaped,
                                    llvm::StringSet<> &seen) {
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return;
  for (Attribute dim : shape.getDims())
    if (auto expr = dyn_cast<ExprAttr>(dim))
      noteExprSymbolNames(expr, seen);
}

// Layout-side syms (offset / storage_size / params) minus `index_syms`
// (bound at access sites) and `shape_syms` (alias dim entries already
// covered).
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

// Sort lex for determinism (StringSet iteration is unordered).
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

// Build `!hc.idx<sym>` per implicit name.
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

// Access-site index expr. Three legal sources:
//   `!hc.idx<expr>` -> expr,
//   `!hc.slice<lower=!hc.idx<expr>, ...>` -> lower,
//   `!hc.slice` no lower -> 0 (Python full-slice).
// Anything else fails: missing sym name to bind to layout's index_sym.
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

// `composeAccessOffsetExpr` lives in `lib/IR/HCAttrs.cpp` -- shared with
// `hc-load-store-to-generic` for hash-cons alignment.

// Materialize composed offset as `!hc.idx<offset_expr>` via
// `hc.idx_apply`. Bound syms listed lex; unbound resolved ambiently.
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

// Buffer with missing layout is a frontend bug (default strided
// layout always attached) -- fail rather than emit `0` over wrong dims.
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

// Bind one SSA per implicit sym from aux expansion. Mismatch fails:
// partially-converted operand range is upstream's bug.
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

// Bare sym name pinned by `!hc.idx<sym>`, else empty. Composite exprs
// fail: value-as-binding only when type's expr IS the symbol leaf.
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

// Bind each bare-sym idx index. Doesn't overwrite aux-supplied bindings.
static void bindBareSymbolIndexOperands(OperandRange indices, sym::Store &store,
                                        llvm::StringMap<Value> &bindings) {
  for (Value idx : indices) {
    StringRef name = pinsBareIdxSymbol(idx.getType(), store);
    if (name.empty())
      continue;
    bindings.try_emplace(name, idx);
  }
}

// Bind ancestor bare-sym block args (e.g. `hc.for_range` IV typed
// `!hc.idx<"$join0">`). Must bind here: for_range->scf.for rewrite
// strips the sym type before the inner apply lowers.
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

// True iff shaped is already post-flatten canonical (no layout,
// single-entry shape). Re-expanding would never fixed-point.
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

// 1D shaped form (the leading flat carrier). nullopt when conversion
// declines (e.g. unresolvable storage_size).
static std::optional<Type>
computeFlatShapedType(SymbolicallyShapedTypeInterface shaped) {
  ShapeAttr originalShape = shaped.getSymbolicShape();
  if (!originalShape)
    return std::nullopt;

  MLIRContext *ctx = shaped.getContext();
  LayoutAttr layout = shaped.getSymbolicLayout();

  // Buffers collapse to `[?]`: host owns allocation; extent comes
  // from host descriptor, not from in-IR sym set.
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

// 1-to-N TypeConverter. Shaped value -> (flat carrier, aux idx per
// implicit sym). Tuples / function types recurse via convertTypes.
// Identity registered first (last-wins). Cast materializations
// fold via `reconcile-unrealized-casts` downstream.
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
      // Already-flat: 1-to-1; expansion would never fixed-point.
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

// `hc.as_layout` is cosmetic post-conversion; forward the converted
// operand expansion.
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

// Thread the trailing aux values of one converted operand into
// `bindings` keyed by implicit-sym name (order matches
// `collectImplicitSyms` on the pre-flatten type).
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

// Bare sym name pinned by `!hc.idx<sym>` (e.g. `!hc.idx<"$WG0">`),
// else empty. Composite exprs fail.
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

// Bind every `!hc.idx<sym>` block arg and pre-`op` SSA in ancestor
// scopes. Sources: for_range IVs, gpu.launch block args ($WG*/$WI*/
// $WGS*), kernel-arg-bundle UCC retype aux ($STRIDE_*, dim syms),
// upstream `hc.idx_apply` results. `try_emplace` keeps pre-seeded
// op-local bindings winning over ancestor scans.
static void collectAncestorIdxBindings(sym::Store &store, Operation *op,
                                       llvm::StringMap<Value> &bindings) {
  for (Block *block = op->getBlock(); block;) {
    for (BlockArgument arg : block->getArguments()) {
      StringRef name = typePinsBareSymbol(store, arg.getType());
      if (!name.empty())
        bindings.try_emplace(name, arg);
    }
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

// Trailing aux value per implicit sym of `preFlattenResult`. First
// matching operand binding wins; missing names emit an unlisted
// `hc.idx_apply` pinned to the bare sym.
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

// Leading flat values only; trailing aux flows via binding map.
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

// Compose multi-index access down to 1D base offset. Benefit 2 so
// driver picks before `RetypeAnyHCOp`'s type-only fallback.
template <typename OpT>
struct ComposeAccessOffsetBase : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  ComposeAccessOffsetBase(const TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<OpT>(converter, ctx, /*benefit=*/2) {}
};

// Drain a 1-to-N adapter range to a flat `Value` list; fails when
// any sub-range isn't single-valued.
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

// True iff access still needs composition: non-1D indexing, or 1D
// source still carrying a layout (buffer_view strided-slice residual).
static bool needsAccessOffsetComposition(unsigned indexCount, Type sourceType) {
  if (indexCount != 1)
    return true;
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(sourceType);
  return shaped && shaped.getSymbolicLayout();
}

static FailureOr<SmallVector<Type>>
convertResultTypeOrFailure(const TypeConverter &converter, Type t) {
  SmallVector<Type> result;
  if (failed(converter.convertType(t, result)))
    return failure();
  if (result.empty())
    return failure();
  return result;
}

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

// Flattener cases for `hc.buffer_view`:
//   identity (1D source shape == 1D result shape) -> forward,
//   single-slice strided (one slice + scalar idx rest, no layout) ->
//   compose row-major offset on flat carrier.
// Anything else bails to the catch-all retyper.
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

// Gate: nothing to flatten when source + indices are already 1D;
// strided-slice can't fold on rank-mismatch without a layout.
static bool bufferViewNeedsFlatten(unsigned indexCount, ShapeAttr preShape,
                                   bool hasLayout) {
  size_t dimCount = preShape.getDims().size();
  if (dimCount == indexCount && dimCount <= 1)
    return false;
  if (!hasLayout && dimCount != indexCount)
    return false;
  return true;
}

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

// Identity: flat carrier matches and types are equal. Type equality
// blocks an element-type miscompile from a shape coincidence.
static bool isBufferViewFlatIdentity(const BufferViewFlattenInputs &in) {
  return in.flatSourceShaped.getSymbolicShape() ==
             in.flatResultShaped.getSymbolicShape() &&
         in.flatSource.getType() == in.flatResultType;
}

// Single slice axis among indices; fail on non-idx/non-slice or >1 slice.
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

// Row-major stride at axis k: product of dims after k.
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

// Sum `index * row_stride` over scalar axes (skip slice axis).
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

// Slice triple with Python defaults: lower->0, upper->axis size, step->1.
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

// Flat triple: lower/upper add `slice_X * stride` to base; step is
// `slice.step * stride`.
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

// Same binding shape as `composeAccessBaseOffset`: aux dim/stride
// from source expansion + bare-sym subscripts + ancestor IV block args.
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
                                flatSource, ValueRange{newSlice},
                                /*unit_axes=*/DenseI64ArrayAttr())
      .getResult();
}

struct SymZeroOne {
  sym::ExprHandle zero;
  sym::ExprHandle one;
};

static FailureOr<SymZeroOne> composeZeroOneExprs(sym::Store &store) {
  auto z = sym::composeExprInt(store, 0);
  auto o = sym::composeExprInt(store, 1);
  if (failed(z) || failed(o))
    return failure();
  return SymZeroOne{*z, *o};
}

// Strided-slice flat view: one slice subscript + scalar idx rest,
// composed via row-major stride layout.
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

// Compose `hc.generic` per-operand per-axis offsets into single-entry
// arrays (post-flatten 1D contract). Type-retype rolled in: leading
// flat carrier per operand, trailing aux into result expansions via
// shared sym-name bindings. Bail on rank mismatch / missing layout.
// Layout-less operands fall back to identity layout.
//
// `composed` flips only when the folded offset differs from input --
// rank-1 layout-less operand identity-folds to itself.
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

// Leading flat carrier per range. `requireSingleton` matches the
// `iter_bounds` 1:1 contract.
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

// Free syms in composed offsets minus iter syms (stay body-scoped).
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

// Body applies can reference ambient syms directly (anything not on
// apply's `symbols` list is iter-scoped or ambient). Surface those
// so `ambient_idxs` carries SSA for the body lowering to find.
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

// Carry over existing ambient bindings; prior passes may have populated.
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

// Lex-sort ambient names for determinism, pin resolved bindings;
// unbound stay free for downstream ambient-context resolution.
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

// Per-result (flat leading type, expansion width). Empty conversion fails.
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

// No-op gate; without it the driver loops forever.
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

// Copy discardable attrs (location hints etc.); the named ones the
// builder wrote already match.
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

// Pair flat result with aux values; reuse binding SSA where available
// instead of materialising fresh ambient applies.
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

// Sym->SSA map + ambient operand list for the new op: operand
// expansions, ancestor block-arg bindings, ambient syms transitively
// needed by offsets and body applies, pre-existing ambient bindings.
static void buildGenericBindingsAndAmbient(
    HCGenericOp op, HCGenericOpOneToNAdaptor adaptor,
    ArrayRef<Attribute> insOffsets, ArrayRef<Attribute> outsOffsets,
    ConversionPatternRewriter &rewriter, llvm::StringMap<Value> &bindings,
    SmallVectorImpl<Value> &ambientIdxsVec,
    SmallVectorImpl<Attribute> &ambientSymsVec) {
  for (auto [orig, range] : llvm::zip_equal(op.getIns(), adaptor.getIns()))
    noteOperandBindings(orig.getType(), range, bindings);
  for (auto [orig, range] : llvm::zip_equal(op.getOuts(), adaptor.getOuts()))
    noteOperandBindings(orig.getType(), range, bindings);

  // Capture ambient sym->SSA now: for_range->scf.for later strips the
  // `!hc.idx<sym>` payload off the IV but `ambient_idxs` already
  // holds the SSA edge by then.
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

// Flat 1D result type of an allocator op; fails on no-op rebuild.
static FailureOr<Type> flatAllocOpResultType(const TypeConverter &converter,
                                             Type origResultType) {
  SmallVector<Type> convertedResults;
  if (failed(converter.convertType(origResultType, convertedResults)) ||
      convertedResults.empty())
    return failure();
  Type flatResult = convertedResults.front();
  if (flatResult == origResultType)
    return failure();
  auto flatShaped = dyn_cast<SymbolicallyShapedTypeInterface>(flatResult);
  if (!flatShaped)
    return failure();
  ShapeAttr flatShape = flatShaped.getSymbolicShape();
  if (!flatShape || flatShape.getDims().empty())
    return failure();
  return flatResult;
}

// Rank-matching shape tuple from flat shape: each dim becomes an
// empty-binding `hc.idx_apply` pinned to the dim expr.
static FailureOr<Value>
buildFlatAllocShapeTuple(ConversionPatternRewriter &rewriter, Location loc,
                         ShapeAttr flatShape) {
  MLIRContext *ctx = rewriter.getContext();
  ArrayAttr emptySymbols = rewriter.getArrayAttr({});
  SmallVector<Value> dimValues;
  SmallVector<Type> dimTypes;
  dimValues.reserve(flatShape.getDims().size());
  dimTypes.reserve(flatShape.getDims().size());
  for (Attribute dim : flatShape.getDims()) {
    auto expr = dyn_cast<ExprAttr>(dim);
    if (!expr)
      return failure();
    Type idxTy = IdxType::get(ctx, expr);
    Value v =
        HCIdxApplyOp::create(rewriter, loc, idxTy, ValueRange{}, emptySymbols)
            .getResult();
    dimValues.push_back(v);
    dimTypes.push_back(idxTy);
  }
  return HCTupleOp::create(rewriter, loc, TupleType::get(ctx, dimTypes),
                           dimValues)
      .getResult();
}

static LogicalResult
spliceFlatAllocShapeOperand(SmallVectorImpl<Value> &flatOperands,
                            HCStaticShapeOpInterface shapeOp, Value newShape) {
  Operation *op = shapeOp.getOperation();
  Value shapeOperand = shapeOp.getStaticShapeOperand();
  auto opOperands = op->getOperands();
  auto shapeIt = llvm::find(opOperands, shapeOperand);
  if (shapeIt == opOperands.end())
    return failure();
  unsigned shapeIdx = std::distance(opOperands.begin(), shapeIt);
  if (shapeIdx >= flatOperands.size())
    return failure();
  flatOperands[shapeIdx] = newShape;
  return success();
}

// Rebuild shape tuple on nullary/fill allocators (zeros/ones/empty/
// full + vec twins) so tuple arity matches the collapsed 1D result.
struct RebuildShapedAllocOpShape
    : public OpInterfaceConversionPattern<HCStaticShapeOpInterface> {
  using Base = OpInterfaceConversionPattern<HCStaticShapeOpInterface>;
  RebuildShapedAllocOpShape(const TypeConverter &converter, MLIRContext *ctx)
      : Base(converter, ctx, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(HCStaticShapeOpInterface shapeOp,
                  ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = shapeOp.getOperation();
    if (shapeOp.getStaticShapeSourceOperand() || op->getNumResults() != 1)
      return failure();

    Type origResultType = op->getResult(0).getType();
    FailureOr<Type> flatResult =
        flatAllocOpResultType(*getTypeConverter(), origResultType);
    if (failed(flatResult))
      return failure();

    Location loc = op->getLoc();
    auto flatShape =
        cast<SymbolicallyShapedTypeInterface>(*flatResult).getSymbolicShape();
    FailureOr<Value> newShape =
        buildFlatAllocShapeTuple(rewriter, loc, flatShape);
    if (failed(newShape))
      return failure();

    SmallVector<Value> flatOperands = flatOperandsOnly(operands);
    if (failed(spliceFlatAllocShapeOperand(flatOperands, shapeOp, *newShape)))
      return failure();

    OperationState state(loc, op->getName());
    state.addOperands(flatOperands);
    state.addTypes(*flatResult);
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);

    llvm::StringMap<Value> bindings;
    for (auto [origOperand, range] :
         llvm::zip_equal(op->getOperands(), operands))
      noteOperandBindings(origOperand.getType(), range, bindings);
    auto auxValues =
        resolveResultAuxValues(rewriter, loc, origResultType, bindings);
    if (failed(auxValues))
      return failure();
    SmallVector<Value> replacement = {newOp->getResult(0)};
    llvm::append_range(replacement, *auxValues);
    SmallVector<ValueRange> replacements = {replacement};
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

// Type-only rebuild for HC dialect ops the populators miss. Attrs +
// regions carry over unchanged. Scoped to `hc` dialect.
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

  // No-op gate; driver loops forever without it.
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

  // Leading flat type per 1-to-N expansion; zero-width is hard failure.
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

  // Build replacement with flat operands/results. Re-runs converter
  // on each region body: `hc.for_range` iter-arg block args must
  // match converted `iter_inits` or the parent verifier rejects.
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

  // Pair flat result with aux; source from operand bindings before
  // emitting fresh ambient applies.
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

// Retype `hc.intrinsic` / `hc.func` / `hc.kernel` signatures + body
// block args. These don't implement `FunctionOpInterface` so upstream
// populator misses them. In production only `hc.intrinsic` reaches
// here -- `hc.kernel` / `hc.func` are consumed by earlier passes
// (`hc-lower-kernels-to-gpu-launch`, `hc-inline-helpers`).
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

    Region &body = op.getBody();
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(newType);
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

// Operand rank for parity check; nullopt for non-rank-constrained
// (`hc.undef`, non-shaped scalars).
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

// Legality: offset count must match operand rank or driver forces
// back to `ComposeGenericOffsets`.
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

// HC symbol-carrier signature legality. `converter.isLegal` only
// inspects op operand/result types (zero on these ops). nullopt for
// non-symbol-carriers.
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

// Top-level legality. Composes function-iface / func.return / call /
// hc-symbol / hc-generic-parity / hc-dialect rules. Default: legal.
static bool isFlattenLegalAtPassBoundary(Operation *op,
                                         const TypeConverter &converter,
                                         MLIRContext *ctx) {
  if (auto fn = dyn_cast<FunctionOpInterface>(op))
    if (auto fnType = dyn_cast<FunctionType>(fn.getFunctionType()))
      return converter.isSignatureLegal(fnType);
  if (isa<func::ReturnOp, func::CallOp>(op))
    return converter.isLegal(op);
  // `hc.as_layout` always illegal -- rewriter drops unconditionally.
  if (isa<HCAsLayoutOp>(op))
    return false;
  if (std::optional<bool> sigLegal = hcSymbolSignatureLegality(op, converter))
    return *sigLegal;
  if (auto generic = dyn_cast<HCGenericOp>(op))
    return isHCGenericLegalAtFlattenBoundary(generic, converter);
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

    // Per-access patterns first by intent; driver picks via benefit
    // (2 vs the generic retype's 1).
    patterns.add<ComposeLoadOffsets, ComposeVLoadOffsets, ComposeStoreOffsets,
                 ComposeGenericOffsets, ComposeBufferViewOffsets, DropAsLayout,
                 RebuildShapedAllocOpShape, RetypeAnyHCOp,
                 ConvertHCSymbolSignatureOp<HCIntrinsicOp>,
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
