// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-elementwise-to-generic`: rewrite the per-element
// shaped arith / cmp / astype ops into the body-driven `hc.generic`
// surface so the post-flatten pipeline only has to lower one shaped
// compute op. Pure source-level rewrite. See the pass description in
// `include/hc/Transforms/Passes.td` and the design in
// `doc/layouts.md`.

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

// Pull the (rank, dims) pair off a shaped HC type. Returns failure
// for `!hc.undef` and for shapes that carry non-`#hc.expr` dim
// entries — either case means the rewriter has nothing to thread
// into the iter-sym layout and the op stays for downstream
// diagnostics.
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

// Body block-arg element type for a shaped operand. Mirrors the
// helper in `lib/IR/HCOps.cpp` (`genericOperandElement`) for the
// shaped cases this rewriter actually emits.
static Type elementType(Type t) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t))
    return shaped.getSymbolicElementType();
  return {};
}

// Synthesise an undef-typed iter bound. `hc-infer-generic-bounds`
// resolves these from operand shapes via identity-offset matching;
// every operand we emit carries identity offsets, so resolution is
// always the operand-shape lookup at the matching axis.
static Value emitUndefBound(OpBuilder &builder, Location loc) {
  return HCUndefValueOp::create(builder, loc,
                                UndefType::get(builder.getContext()));
}

// Build the iter-sym attr list (`["i_0", "i_1", ...]`) and the
// matching all-parallel iter-kinds attr.
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

// Identity per-axis offset array for the given iter syms. Reused
// for every operand the rewrite emits, both ins and outs.
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

// Per-operand offsets that project broadcast-unit axes to literal 0.
// `operandShape` and `resultShape` are rank-aligned; the axes where
// `operandShape` carries a literal `1` (and the result doesn't) read
// the same element on every iteration, so the offset is `0`; the
// other axes carry the matching iter sym in identity form. The
// existing bounds-inference (`-hc-infer-generic-bounds`) keys off
// identity offsets only, so the `0` projection is transparent to it.
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

// Build the tuple<idx, ...> shape SSA the value-init op (`hc.zeros`
// / `hc.vzeros`) requires. The init is overwritten by the body on
// every iteration, so the dim values are just placeholders shaped
// like the result; reusing the same `hc.undef_value` instances
// keeps the IR small and lets DCE drop the duplicate later.
static Value buildShapeTuple(OpBuilder &builder, Location loc,
                             ValueRange dims) {
  SmallVector<Type> elemTypes(
      llvm::map_range(dims, [](Value v) -> Type { return v.getType(); }));
  auto tupleTy = TupleType::get(builder.getContext(), elemTypes);
  return HCTupleOp::create(builder, loc, tupleTy, dims);
}

// `hc.zeros` for tensor / bare_tensor results, `hc.vzeros` for
// vector / bare_vector. Other shaped flavours don't appear at this
// pre-flatten stage.
static Value emitInit(OpBuilder &builder, Location loc, Type resultTy,
                      Value shape) {
  if (isa<mlir::hc::VectorType, BareVectorType>(resultTy))
    return HCVZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                              /*layout=*/LayoutAttr{});
  return HCZerosOp::create(builder, loc, resultTy, shape, TypeAttr(),
                           /*layout=*/LayoutAttr{});
}

// Spec for one rewrite. Caller fills it in per source op; the
// `emitElementwise` helper assembles the `hc.generic` from this
// uniform shape so the per-op visitors stay tiny.
struct ElementwiseSpec {
  // Shaped input operands the new op carries as `ins`. Order is
  // preserved into the body block-arg list.
  SmallVector<Value> shapedIns;
  // Result type — also the type of the synthesised init operand.
  Type resultTy;
  // Per-axis dim expressions of the result; the rewriter emits one
  // iter sym per entry.
  SmallVector<ExprAttr> resultShape;
  // Per-operand symbolic shape, rank-aligned with `resultShape`. Used
  // to project broadcast-unit axes (operand dim `1` against a
  // non-`1` result dim) to a literal-`0` offset in the offset array.
  // Same-shape operands get identity offsets out of this projection
  // — broadcast and non-broadcast share one path.
  SmallVector<SmallVector<ExprAttr>> operandShapes;
  // Body builder. `insArgs` is one block arg per entry of
  // `shapedIns`, in the same order; `outArg` is the carry block arg
  // (unused for all-parallel iters but still part of the block); the
  // builder returns the scalar to yield.
  std::function<Value(OpBuilder &, Location, ValueRange /*insArgs*/,
                      Value /*outArg*/, Type /*resElemTy*/)>
      bodyBuilder;
};

// Common assembler. Replaces the original op with the synthesised
// `hc.generic`, threading the spec's body builder through.
static LogicalResult emitElementwise(Operation *op, sym::Store &store,
                                     ElementwiseSpec spec) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  // Iter bounds use undef placeholders; bound inference resolves
  // them from the operand shapes via identity-offset matching. A
  // broadcast operand only contributes identity bindings for the
  // axes it doesn't project — for the unit-broadcast axes the init's
  // identity offsets carry the inference, which is why the spec
  // requires every result dim to appear somewhere.
  size_t rank = spec.resultShape.size();
  SmallVector<Value> iterBounds;
  iterBounds.reserve(rank);
  for (size_t k = 0; k < rank; ++k)
    iterBounds.push_back(emitUndefBound(builder, loc));

  IterMeta iter = buildIterMeta(ctx, rank);
  ArrayAttr identity = identityOffsetArray(ctx, store, iter.syms);

  // The init's shape tuple needs SSA dim values. Reuse the iter
  // bounds — they're undef-typed today; bound inference replaces
  // them with empty-binding `hc.idx_apply` ops once we know the
  // dim expressions, which the init shape then picks up uniformly.
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

// Broadcast-aware operand-shape collector. Result and operand ranks
// must agree; per-axis the operand may either match the result dim
// or carry a literal `1` (broadcast). Mismatched non-`1` dims fall
// through to a downstream diagnostic. Returns the per-operand shape
// list rank-aligned with the result so the emitter can build per-
// operand offset arrays.
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

// Binary arith / boolean: `hc.add`, `hc.sub`, `hc.mul`, `hc.div`,
// `hc.mod`, `hc.and`, `hc.or`. The body op is the same op kind as
// the source — these are element-type-generic at the dialect level
// and lower to the matching `arith` later. Operands may broadcast
// against the result via unit-size axes; `matchBroadcastShapes`
// gates on that and threads the per-operand shape through to the
// emitter so the resulting `hc.generic` carries per-operand offset
// arrays that fold the unit dims to a literal `0`.
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

// Unary arith / boolean: `hc.neg`, `hc.not`. Same per-element body
// shape as the binary case with one block arg.
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

// Comparisons. Result element type differs from input element type
// (`i1` / `!hc.pred`), so the body returns a separate scalar; the
// shape-gating helper (`matchBroadcastShapes`) only checks dims —
// element types stay independent and the body builder spells the
// result element out.
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

// `hc.astype value, target = T : in -> out`. Element type changes,
// shape is preserved. Body emits a scalar `hc.astype` to the result
// element type so the lowering doesn't have to special-case typed
// vs untyped block args.
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

    // Collect first; mutate after. `op->erase()` inside the walk
    // would invalidate the iterator. Per-class buckets keep the
    // dispatch simple — the per-op visitors are templated on the
    // exact op type so we can't drive them through a single virtual
    // call.
    SmallVector<Operation *> toRewrite;
    root->walk([&](Operation *op) {
      if (isa<HCAddOp, HCSubOp, HCMulOp, HCDivOp, HCModOp, HCAndOp, HCOrOp,
              HCNegOp, HCNotOp, HCCmpLtOp, HCCmpLeOp, HCCmpGtOp, HCCmpGeOp,
              HCCmpEqOp, HCCmpNeOp, HCAsTypeOp>(op))
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
