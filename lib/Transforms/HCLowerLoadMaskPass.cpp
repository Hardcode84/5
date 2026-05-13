// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-load-mask`: rewrite every `hc.load_mask` against a
// kernel-arg bundle or a static bare tensor into `hc.mask_from_sizes` whose
// per-axis sizes are computed from the multi-dim source. See the pass
// description in `include/hc/Transforms/Passes.td`.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLOADMASK
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// The kernel-arg ABI fragment planted by `hc-lower-kernels-to-gpu-launch`:
// a single UCC whose inputs are `(ptr, dim_0, ..., dim_{n-1}, stride_0, ...,
// stride_{n-1})` and whose single output is the multi-dim `!hc.buffer`.
// `flattenKernelArgDims` walks back through the UCC to recover the per-axis
// dim SSA values. Returns failure for sources that aren't shaped this way
// (workgroup-staged bare tensors, raw ptrs, post-flatten retypes — pre-flatten
// none of those should appear under `hc.load_mask`, but we fail soft so the
// op stays for `hc-lower-launch-body`'s legacy handler to diagnose).
static FailureOr<SmallVector<Value>> kernelArgDims(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getOutputs().size() != 1 || cast.getInputs().size() < 1)
    return failure();
  // Expecting one ptr input plus 2*rank index inputs.
  unsigned inputs = cast.getInputs().size();
  if ((inputs - 1) % 2 != 0)
    return failure();
  unsigned rank = (inputs - 1) / 2;
  SmallVector<Value> dims;
  dims.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value dim = cast.getInputs()[1 + axis];
    if (!dim.getType().isIndex())
      return failure();
    dims.push_back(dim);
  }
  return dims;
}

// Bridge an `!hc.idx<expr>` value into `index` via an UCC. Pre-flatten the
// slice subscript SSA values still ride on `!hc.idx`; `hc-lower-launch-body`
// later folds the matching `idx → index` UCC pair planted by
// `ConvertIdxApplyOp` against ours, so the cast doesn't survive into the
// lowered body.
static Value castIdxToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), value)
      .getResult(0);
}

// Mask size for one slice axis: `ceildiv(extent - lower, step)` when `step`
// isn't unit, falling back to `extent - lower` for the unit-stride case so
// post-canonicalize folds the trivial `(... + 0) / 1` away.
static Value axisMaskSize(OpBuilder &builder, Location loc, Value extent,
                          Value lower, Value step) {
  Value remaining = arith::SubIOp::create(builder, loc, extent, lower);
  APInt stepConst;
  bool unitStride = step && matchPattern(step, m_ConstantInt(&stepConst)) &&
                    stepConst.getSExtValue() == 1;
  if (!step || unitStride)
    return remaining;
  Value one = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  Value stepMinusOne = arith::SubIOp::create(builder, loc, step, one);
  Value adjusted = arith::AddIOp::create(builder, loc, remaining, stepMinusOne);
  return arith::DivSIOp::create(builder, loc, adjusted, step).getResult();
}

// Build the per-axis mask size for one `(axisExtent, indexValue)` pair.
// Returns the size if the index is a slice; returns null and sets
// `failed` true on the kinds of indices we deliberately punt on (non-slice
// idx subscripts — the corresponding axis collapses out of the result
// shape and doesn't contribute a mask size). `failed` is left untouched for
// well-formed slices.
static Value sliceAxisMaskSize(OpBuilder &builder, Location loc,
                               Value axisExtent, Value indexValue) {
  auto slice = indexValue.getDefiningOp<HCSliceExprOp>();
  if (!slice)
    return {};
  Value lower = slice.getLower();
  Value step = slice.getStep();
  Value lowerIndex =
      lower ? castIdxToIndex(builder, loc, lower)
            : arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value stepIndex = step ? castIdxToIndex(builder, loc, step) : Value{};
  return axisMaskSize(builder, loc, axisExtent, lowerIndex, stepIndex);
}

// Per-axis static extents from a static-shape bare tensor source. The
// launch-body's legacy handler reads these too; we surface them here so the
// staging-tile path can flow through the same `hc.mask_from_sizes` machinery
// as the kernel-arg path.
static FailureOr<SmallVector<Value>>
staticBareTensorExtents(OpBuilder &builder, Location loc, Type type) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return failure();
  ShapeAttr shape = shaped.getSymbolicShape();
  SmallVector<Value> dims;
  dims.reserve(shape.getDims().size());
  for (Attribute attr : shape.getDims()) {
    auto expr = dyn_cast<ExprAttr>(attr);
    if (!expr)
      return failure();
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()));
    if (!value || *value < 0)
      return failure();
    dims.push_back(
        arith::ConstantIndexOp::create(builder, loc, *value).getResult());
  }
  return dims;
}

// Try to derive the per-axis extents from `source`. Kernel-arg bundles come
// first because that's the dominant shape in lowered launch bodies; static
// bare tensors are the staging-tile / LDS fallback. Anything else (raw
// `!hc.ptr` sources, post-flatten retypes that already collapsed the
// per-axis structure) returns failure and leaves the `hc.load_mask` for the
// downstream handler.
static FailureOr<SmallVector<Value>>
resolveSourceExtents(OpBuilder &builder, Location loc, Value source) {
  if (FailureOr<SmallVector<Value>> dims = kernelArgDims(source);
      succeeded(dims))
    return *dims;
  return staticBareTensorExtents(builder, loc, source.getType());
}

// Static rank-N shape from the mask result type. The legacy
// `hc.load_mask` carries it implicitly on the bare result; the new op
// pins it as an attribute so flatten's 1D retype doesn't strip the
// per-axis bounds.
static FailureOr<SmallVector<int64_t>> resultStaticShape(Type type) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return failure();
  ShapeAttr shape = shaped.getSymbolicShape();
  SmallVector<int64_t> dims;
  dims.reserve(shape.getDims().size());
  for (Attribute attr : shape.getDims()) {
    auto expr = dyn_cast<ExprAttr>(attr);
    if (!expr)
      return failure();
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()));
    if (!value || *value < 0)
      return failure();
    dims.push_back(*value);
  }
  return dims;
}

static LogicalResult lowerLoadMask(HCLoadMaskOp op) {
  OpBuilder builder(op);
  FailureOr<SmallVector<Value>> extents =
      resolveSourceExtents(builder, op.getLoc(), op.getSource());
  if (failed(extents))
    return success(); // Leave the op for the legacy launch-body handler.
  if (extents->size() != op.getIndices().size())
    return success();

  SmallVector<Value> maskSizes;
  maskSizes.reserve(op.getIndices().size());
  for (auto [axisIdx, indexValue] : llvm::enumerate(op.getIndices())) {
    if (isa<SliceType>(indexValue.getType())) {
      Value size = sliceAxisMaskSize(builder, op.getLoc(), (*extents)[axisIdx],
                                     indexValue);
      if (!size)
        return success(); // Non-canonical slice; let the legacy path handle it.
      maskSizes.push_back(size);
    }
    // Non-slice subscripts (plain `!hc.idx`) collapse the axis out of the
    // result shape, so they don't contribute a per-axis size.
  }

  FailureOr<SmallVector<int64_t>> shape =
      resultStaticShape(op.getMask().getType());
  if (failed(shape) || shape->size() != maskSizes.size())
    return success();

  auto replacement = HCMaskFromSizesOp::create(
      builder, op.getLoc(), op.getMask().getType(), maskSizes,
      builder.getDenseI64ArrayAttr(*shape));
  op.replaceAllUsesWith(replacement.getOperation());
  op.erase();
  return success();
}

struct HCLowerLoadMaskPass
    : public hc::impl::HCLowerLoadMaskBase<HCLowerLoadMaskPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<HCLoadMaskOp> targets;
    getOperation()->walk([&](HCLoadMaskOp op) { targets.push_back(op); });
    for (HCLoadMaskOp op : targets)
      if (failed(lowerLoadMask(op)))
        return signalPassFailure();
  }
};

} // namespace
