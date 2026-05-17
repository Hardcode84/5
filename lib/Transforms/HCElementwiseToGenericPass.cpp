// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-elementwise-to-generic`: rewrite per-element shaped
// arith / cmp / astype into `hc.generic`. See `doc/layouts.md`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCELEMENTWISETOGENERIC
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Failure on `!hc.undef` or non-`#hc.expr` dims; leaves op for diagnostics.
static FailureOr<SmallVector<ExprAttr>> getOperandShape(Type t) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t);
  if (!shaped)
    return failure();
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return failure();
  SmallVector<ExprAttr> dims;
  dims.reserve(shape.getDims().size());
  for (Attribute a : shape.getDims()) {
    auto e = dyn_cast<ExprAttr>(a);
    if (!e)
      return failure();
    dims.push_back(e);
  }
  return dims;
}

// Body block-arg element type for a shaped operand.
static Type elementType(Type t) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t))
    return shaped.getSymbolicElementType();
  return {};
}

// Undef placeholder — bounds inference resolves via identity-offset matching.
static Value emitUndefBound(OpBuilder &builder, Location loc) {
  return HCUndefValueOp::create(builder, loc,
                                UndefType::get(builder.getContext()));
}

// `i_0`..`i_{r-1}` iter syms, all-parallel kinds.
struct IterMeta {
  SmallVector<StringAttr> syms;
  ArrayAttr symsAttr;
  ArrayAttr kindsAttr;
};

static IterMeta buildIterMeta(MLIRContext *ctx, size_t rank) {
  IterMeta out;
  out.syms.reserve(rank);
  SmallVector<Attribute> symAttrs;
  symAttrs.reserve(rank);
  SmallVector<Attribute> kindAttrs;
  kindAttrs.reserve(rank);
  for (size_t k = 0; k < rank; ++k) {
    auto sym = StringAttr::get(ctx, ("i_" + Twine(k)).str());
    out.syms.push_back(sym);
    symAttrs.push_back(sym);
    kindAttrs.push_back(IterKindAttr::get(ctx, IterKind::Parallel));
  }
  out.symsAttr = ArrayAttr::get(ctx, symAttrs);
  out.kindsAttr = ArrayAttr::get(ctx, kindAttrs);
  return out;
}

// Identity offsets; reused on ins and outs.
static ArrayAttr identityOffsetArray(MLIRContext *ctx, sym::Store &store,
                                     ArrayRef<StringAttr> iterSyms) {
  SmallVector<Attribute> exprs;
  exprs.reserve(iterSyms.size());
  for (StringAttr name : iterSyms) {
    auto handle = sym::composeExprSym(store, name.getValue());
    assert(succeeded(handle) && "iter sym name must compose to an expression");
    exprs.push_back(ExprAttr::get(ctx, *handle));
  }
  return ArrayAttr::get(ctx, exprs);
}

// Broadcast-unit axes → `0` offset; identity elsewhere. Bounds inference
// keys off identity only, so the `0` projection is transparent.
static FailureOr<ArrayAttr> broadcastOperandOffsetArray(
    MLIRContext *ctx, sym::Store &store, ArrayRef<StringAttr> iterSyms,
    ArrayRef<ExprAttr> operandShape, ArrayRef<ExprAttr> resultShape) {
  assert(operandShape.size() == iterSyms.size() &&
         resultShape.size() == iterSyms.size() &&
         "operand and iter sym ranks must match the result");
  auto oneHandle = sym::composeExprInt(store, 1);
  if (failed(oneHandle))
    return failure();
  auto zeroHandle = sym::composeExprInt(store, 0);
  if (failed(zeroHandle))
    return failure();
  ExprAttr oneAttr = ExprAttr::get(ctx, *oneHandle);
  ExprAttr zeroAttr = ExprAttr::get(ctx, *zeroHandle);
  SmallVector<Attribute> exprs;
  exprs.reserve(iterSyms.size());
  for (auto [iterSym, opDim, resDim] :
       llvm::zip_equal(iterSyms, operandShape, resultShape)) {
    if (opDim == oneAttr && resDim != oneAttr) {
      exprs.push_back(zeroAttr);
      continue;
    }
    auto handle = sym::composeExprSym(store, iterSym.getValue());
    if (failed(handle))
      return failure();
    exprs.push_back(ExprAttr::get(ctx, *handle));
  }
  return ArrayAttr::get(ctx, exprs);
}

// Init shape placeholders — init is overwritten every iter; DCE drops dupes.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// `hc.zeros` / `hc.vzeros` per flavor; semantic carriers rejected upstream.
static Value emitInit(OpBuilder &builder, Location loc, Type resultTy,
                      Value shape) {
  if (isa<BareVectorType>(resultTy))
    return HCVZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                              /*layout=*/LayoutAttr{});
  assert(isa<BareTensorType>(resultTy) &&
         "elementwise-to-generic result must be a bare shaped carrier");
  return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                           /*layout=*/LayoutAttr{});
}

// Spec for one rewrite; `emitElementwise` assembles the generic.
struct ElementwiseSpec {
  // Order preserved into the body block-arg list.
  SmallVector<Value> shapedIns;
  Type resultTy;
  SmallVector<ExprAttr> resultShape;
  // Rank-aligned with `resultShape`; unit dims project to `0` offset.
  SmallVector<SmallVector<ExprAttr>> operandShapes;
  // `outArg` is the carry block arg (unused for all-parallel).
  std::function<Value(OpBuilder &, Location, ValueRange /*insArgs*/,
                      Value /*outArg*/, Type /*resElemTy*/)>
      bodyBuilder;
};

// Replaces `op` with synthesised `hc.generic`.
static LogicalResult emitElementwise(Operation *op, sym::Store &store,
                                     ElementwiseSpec spec) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  // Undef bounds; inference resolves via identity offsets. Init's
  // identity offsets cover broadcast-unit axes.
  size_t rank = spec.resultShape.size();
  SmallVector<Value> iterBounds;
  iterBounds.reserve(rank);
  for (size_t k = 0; k < rank; ++k)
    iterBounds.push_back(emitUndefBound(builder, loc));

  IterMeta iter = buildIterMeta(ctx, rank);
  ArrayAttr identity = identityOffsetArray(ctx, store, iter.syms);

  // Init shape reuses iter-bound undefs; inference rewrites them in place.
  Value shapeTuple = buildShapeTuple(builder, loc, iterBounds);
  Value initOut = emitInit(builder, loc, spec.resultTy, shapeTuple);

  assert(spec.operandShapes.size() == spec.shapedIns.size() &&
         "spec.operandShapes must be aligned with shapedIns");
  SmallVector<Attribute> insOffArr;
  insOffArr.reserve(spec.shapedIns.size());
  for (auto [opShape, _operand] :
       llvm::zip_equal(spec.operandShapes, spec.shapedIns)) {
    FailureOr<ArrayAttr> off = broadcastOperandOffsetArray(
        ctx, store, iter.syms, opShape, spec.resultShape);
    if (failed(off))
      return failure();
    insOffArr.push_back(*off);
  }
  ArrayAttr insOffsets = ArrayAttr::get(ctx, insOffArr);
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {identity});

  SmallVector<Value> insVals(spec.shapedIns);
  SmallVector<Value> outsVals{initOut};
  auto generic = HCGenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{spec.resultTy}, iter.symsAttr,
      ValueRange(iterBounds), iter.kindsAttr, ValueRange(insVals),
      ValueRange(outsVals), /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  SmallVector<Value> insArgs;
  insArgs.reserve(spec.shapedIns.size());
  for (Value in : spec.shapedIns) {
    Type e = elementType(in.getType());
    insArgs.push_back(body->addArgument(e, loc));
  }
  Type resElem = elementType(spec.resultTy);
  Value outArg = body->addArgument(resElem, loc);
  generic.getBody().push_back(body);

  OpBuilder bodyBuilder(body, body->begin());
  Value yielded = spec.bodyBuilder(bodyBuilder, loc, insArgs, outArg, resElem);
  HCYieldOp::create(bodyBuilder, loc, ValueRange{yielded});

  op->replaceAllUsesWith(generic.getResults());
  op->erase();
  return success();
}

// Operand rank = result rank; per-axis match or operand-`1` only.
static FailureOr<SmallVector<SmallVector<ExprAttr>>>
matchBroadcastShapes(MLIRContext *ctx, sym::Store &store,
                     ArrayRef<ExprAttr> resultShape, ValueRange ins) {
  auto oneHandle = sym::composeExprInt(store, 1);
  if (failed(oneHandle))
    return failure();
  ExprAttr oneAttr = ExprAttr::get(ctx, *oneHandle);
  SmallVector<SmallVector<ExprAttr>> operandShapes;
  operandShapes.reserve(ins.size());
  for (Value in : ins) {
    auto inShape = getOperandShape(in.getType());
    if (failed(inShape))
      return failure();
    if (inShape->size() != resultShape.size())
      return failure();
    for (auto [opDim, resDim] : llvm::zip_equal(*inShape, resultShape)) {
      if (opDim == resDim)
        continue;
      if (opDim == oneAttr)
        continue;
      return failure();
    }
    operandShapes.push_back(std::move(*inShape));
  }
  return operandShapes;
}

// ----- per-op visitors --------------------------------------------------

// Binary arith / boolean. Body op = source op kind.
template <typename OpT>
static LogicalResult rewriteBinaryHomogeneous(OpT op, sym::Store &store) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Type resultTy = op.getResult().getType();
  auto shape = getOperandShape(resultTy);
  if (failed(shape))
    return failure();
  auto operandShapes =
      matchBroadcastShapes(op.getContext(), store, *shape, {lhs, rhs});
  if (failed(operandShapes))
    return failure();
  ElementwiseSpec spec;
  spec.shapedIns = {lhs, rhs};
  spec.resultTy = resultTy;
  spec.resultShape = std::move(*shape);
  spec.operandShapes = std::move(*operandShapes);
  spec.bodyBuilder = [](OpBuilder &b, Location loc, ValueRange args,
                        Value /*out*/, Type elem) -> Value {
    return OpT::create(b, loc, elem, args[0], args[1]).getResult();
  };
  return emitElementwise(op, store, spec);
}

// Unary arith / boolean: `hc.neg`, `hc.not`.
template <typename OpT>
static LogicalResult rewriteUnaryHomogeneous(OpT op, sym::Store &store) {
  Value v = op.getValue();
  Type resultTy = op.getResult().getType();
  auto shape = getOperandShape(resultTy);
  if (failed(shape))
    return failure();
  auto operandShapes =
      matchBroadcastShapes(op.getContext(), store, *shape, {v});
  if (failed(operandShapes))
    return failure();
  ElementwiseSpec spec;
  spec.shapedIns = {v};
  spec.resultTy = resultTy;
  spec.resultShape = std::move(*shape);
  spec.operandShapes = std::move(*operandShapes);
  spec.bodyBuilder = [](OpBuilder &b, Location loc, ValueRange args,
                        Value /*out*/, Type elem) -> Value {
    return OpT::create(b, loc, elem, args[0]).getResult();
  };
  return emitElementwise(op, store, spec);
}

// `hc.builtin_call` elementwise (sqrt, exp, ...): re-emit scalar call on body
// args.
static LogicalResult rewriteBuiltinCall(HCBuiltinCallOp op, sym::Store &store) {
  OperandRange args = op.getArgs();
  if (args.empty())
    return failure();
  Type resultTy = op.getResult().getType();
  auto shape = getOperandShape(resultTy);
  if (failed(shape))
    return failure();
  auto operandShapes =
      matchBroadcastShapes(op.getContext(), store, *shape, args);
  if (failed(operandShapes))
    return failure();
  ElementwiseSpec spec;
  spec.shapedIns.assign(args.begin(), args.end());
  spec.resultTy = resultTy;
  spec.resultShape = std::move(*shape);
  spec.operandShapes = std::move(*operandShapes);
  StringAttr name = op.getNameAttr();
  spec.bodyBuilder = [name](OpBuilder &b, Location loc, ValueRange callArgs,
                            Value /*out*/, Type elem) -> Value {
    return HCBuiltinCallOp::create(b, loc, elem, name, callArgs).getResult();
  };
  return emitElementwise(op, store, spec);
}

// Comparisons: result elem (`i1`/`!hc.pred`) differs from input elem.
template <typename OpT>
static LogicalResult rewriteCmp(OpT op, sym::Store &store) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Type resultTy = op.getResult().getType();
  auto resultShape = getOperandShape(resultTy);
  if (failed(resultShape))
    return failure();
  auto operandShapes =
      matchBroadcastShapes(op.getContext(), store, *resultShape, {lhs, rhs});
  if (failed(operandShapes))
    return failure();
  Type lhsElem = elementType(lhs.getType());
  Type rhsElem = elementType(rhs.getType());
  Type resElem = elementType(resultTy);
  if (!lhsElem || !rhsElem || !resElem)
    return failure();
  ElementwiseSpec spec;
  spec.shapedIns = {lhs, rhs};
  spec.resultTy = resultTy;
  spec.resultShape = std::move(*resultShape);
  spec.operandShapes = std::move(*operandShapes);
  spec.bodyBuilder = [resElem](OpBuilder &b, Location loc, ValueRange args,
                               Value /*out*/, Type /*ignored*/) -> Value {
    return OpT::create(b, loc, resElem, args[0], args[1]).getResult();
  };
  return emitElementwise(op, store, spec);
}

// `hc.astype`: shape preserved, elem changes; body emits scalar astype.
static LogicalResult rewriteAsType(HCAsTypeOp op, sym::Store &store) {
  Value v = op.getValue();
  Type resultTy = op.getResult().getType();
  auto resultShape = getOperandShape(resultTy);
  if (failed(resultShape))
    return failure();
  auto operandShapes =
      matchBroadcastShapes(op.getContext(), store, *resultShape, {v});
  if (failed(operandShapes))
    return failure();
  Type resElem = elementType(resultTy);
  if (!resElem)
    return failure();
  ElementwiseSpec spec;
  spec.shapedIns = {v};
  spec.resultTy = resultTy;
  spec.resultShape = std::move(*resultShape);
  spec.operandShapes = std::move(*operandShapes);
  spec.bodyBuilder = [resElem](OpBuilder &b, Location loc, ValueRange args,
                               Value /*out*/, Type /*ignored*/) -> Value {
    return HCAsTypeOp::create(b, loc, resElem, args[0], TypeAttr::get(resElem))
        .getResult();
  };
  return emitElementwise(op, store, spec);
}

struct HCElementwiseToGenericPass
    : public hc::impl::HCElementwiseToGenericBase<HCElementwiseToGenericPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto &store =
        root->getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();

    // Collect before mutate; erase-during-walk invalidates the iterator.
    SmallVector<Operation *> toRewrite;
    root->walk([&](Operation *op) {
      if (isa<HCAddOp, HCSubOp, HCMulOp, HCDivOp, HCModOp, HCAndOp, HCOrOp,
              HCNegOp, HCNotOp, HCBuiltinCallOp, HCCmpLtOp, HCCmpLeOp,
              HCCmpGtOp, HCCmpGeOp, HCCmpEqOp, HCCmpNeOp, HCAsTypeOp>(op))
        toRewrite.push_back(op);
    });

    for (Operation *op : toRewrite) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<HCAddOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCAddOp>(o, store);
              })
              .Case<HCSubOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCSubOp>(o, store);
              })
              .Case<HCMulOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCMulOp>(o, store);
              })
              .Case<HCDivOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCDivOp>(o, store);
              })
              .Case<HCModOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCModOp>(o, store);
              })
              .Case<HCAndOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCAndOp>(o, store);
              })
              .Case<HCOrOp>([&](auto o) {
                return rewriteBinaryHomogeneous<HCOrOp>(o, store);
              })
              .Case<HCNegOp>([&](auto o) {
                return rewriteUnaryHomogeneous<HCNegOp>(o, store);
              })
              .Case<HCNotOp>([&](auto o) {
                return rewriteUnaryHomogeneous<HCNotOp>(o, store);
              })
              .Case<HCBuiltinCallOp>(
                  [&](auto o) { return rewriteBuiltinCall(o, store); })
              .Case<HCCmpLtOp>(
                  [&](auto o) { return rewriteCmp<HCCmpLtOp>(o, store); })
              .Case<HCCmpLeOp>(
                  [&](auto o) { return rewriteCmp<HCCmpLeOp>(o, store); })
              .Case<HCCmpGtOp>(
                  [&](auto o) { return rewriteCmp<HCCmpGtOp>(o, store); })
              .Case<HCCmpGeOp>(
                  [&](auto o) { return rewriteCmp<HCCmpGeOp>(o, store); })
              .Case<HCCmpEqOp>(
                  [&](auto o) { return rewriteCmp<HCCmpEqOp>(o, store); })
              .Case<HCCmpNeOp>(
                  [&](auto o) { return rewriteCmp<HCCmpNeOp>(o, store); })
              .Case<HCAsTypeOp>([&](auto o) { return rewriteAsType(o, store); })
              .Default([](Operation *) { return failure(); });
      (void)result;
    }
  }
};

} // namespace
