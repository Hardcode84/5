// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-verify-static-shapes`, the post-inference gate for ops that
// materialize tensors/vectors from SSA shape carriers. Participating ops expose
// their operands through `HCStaticShapeOpInterface`; shaped source/result types
// expose rank and dimension metadata through `SymbolicallyShapedTypeInterface`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOpsInterfaces.h"
#include "hc/IR/HCTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCVERIFYSTATICSHAPES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static ShapeAttr symbolicShapeFromType(Type type) {
  if (auto shaped = dyn_cast_or_null<SymbolicallyShapedTypeInterface>(type))
    return shaped.getSymbolicShape();
  return {};
}

static LogicalResult verifyResultShape(Operation *op, Type resultType,
                                       ShapeAttr shape, bool vectorResult) {
  // Tensor and vector results have different semantics even though both carry
  // symbolic shapes through the same type interface.
  bool hasExpectedShell = vectorResult ? isa<mlir::hc::VectorType>(resultType)
                                       : isa<mlir::hc::TensorType>(resultType);
  if (!hasExpectedShell)
    return op->emitOpError("expected result type ")
           << (vectorResult ? "!hc.vector" : "!hc.tensor") << ", got "
           << resultType;

  ShapeAttr actual = symbolicShapeFromType(resultType);
  if (actual != shape)
    return op->emitOpError("shape operand ")
           << shape << " does not match result type shape " << actual;
  return success();
}

static LogicalResult verifyIndexStructure(Operation *op, ValueRange indices,
                                          Type sourceType,
                                          ShapeAttr sourceShape) {
  // Indices bind a prefix of the source's index_syms: tile coords for
  // a plain shape, tile coords + selectors for a `#hc.layout` whose
  // `index_syms` is strictly longer than `shape_syms` (see
  // HC_LayoutAttr description). Bind count is `index_syms.size()` if a
  // layout is attached, else the rank of the shape (implicit identity
  // layout). Fewer than the full bind count is legal — the trailing
  // tile dims get filled by the source/result shape.
  size_t maxArity = sourceShape.getDims().size();
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(sourceType))
    if (LayoutAttr layout = shaped.getSymbolicLayout())
      maxArity = layout.getIndexSyms().size();
  if (indices.size() > maxArity) {
    auto diag = op->emitOpError("has ")
                << indices.size() << " index operand(s) for source bind count "
                << maxArity;
    if (maxArity != sourceShape.getDims().size())
      diag.attachNote() << "source layout's index_syms is " << maxArity
                        << " (rank " << sourceShape.getDims().size() << " + "
                        << (maxArity - sourceShape.getDims().size())
                        << " selector(s))";
    return diag;
  }
  for (auto [idx, value] : llvm::enumerate(indices)) {
    Type type = value.getType();
    if (isa<IdxType, SliceType>(type) || type.isIntOrIndex())
      continue;
    return op->emitOpError("index #")
           << idx << " must be !hc.idx, !hc.slice, or builtin integer/index, "
           << "got " << type;
  }
  return success();
}

static LogicalResult verifyStaticShapeOp(HCStaticShapeOpInterface shapeOp) {
  Operation *op = shapeOp.getOperation();
  FailureOr<ShapeAttr> staticShape = verifyStaticShapeFromTupleType(
      shapeOp.getStaticShapeOperand().getType(), op);
  if (failed(staticShape))
    return failure();

  if (Value source = shapeOp.getStaticShapeSourceOperand()) {
    ShapeAttr sourceShapeAttr = symbolicShapeFromType(source.getType());
    if (!sourceShapeAttr)
      return op->emitOpError("source operand type must carry a symbolic shape, "
                             "got ")
             << source.getType();

    if (failed(verifyIndexStructure(op, shapeOp.getStaticShapeIndexOperands(),
                                    source.getType(), sourceShapeAttr)))
      return failure();
  }
  return verifyResultShape(op, shapeOp.getStaticShapedResultType(),
                           *staticShape, shapeOp.hasStaticVectorResult());
}

struct HCVerifyStaticShapesPass
    : public hc::impl::HCVerifyStaticShapesBase<HCVerifyStaticShapesPass> {
  using Base::Base;

  void runOnOperation() override {
    bool foundError = false;
    getOperation()->walk([&](HCStaticShapeOpInterface shapeOp) {
      foundError |= failed(verifyStaticShapeOp(shapeOp));
    });
    if (foundError)
      signalPassFailure();
  }
};

} // namespace

// `createHCVerifyStaticShapesPass()` is emitted by tablegen.
