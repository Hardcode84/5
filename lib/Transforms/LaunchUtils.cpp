// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LaunchUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;
using namespace mlir::hc;

FailureOr<std::pair<Value, Value>>
mlir::hc::linearizedThreadAndSize(OpBuilder &builder, Location loc,
                                  Operation *anchor) {
  auto launch = anchor->getParentOfType<gpu::LaunchOp>();
  if (!launch)
    return failure();
  gpu::KernelDim3 tids = launch.getThreadIds();
  // `getBlockSizeOperandValues` reaches the values defined above the launch
  // (the operands that pin the block shape). Inside the body they're still
  // dominating SSA values, and using the outer form keeps the cooperative
  // loop's IR close to the values the materializeBoundExpr lowering already
  // surfaces under the `$WGS*` symbols.
  gpu::KernelDim3 sizes = launch.getBlockSizeOperandValues();
  Value tzBy = arith::MulIOp::create(builder, loc, tids.z, sizes.y);
  Value tzByPlusTy = arith::AddIOp::create(builder, loc, tzBy, tids.y);
  Value rowSpan = arith::MulIOp::create(builder, loc, tzByPlusTy, sizes.x);
  Value linearTid = arith::AddIOp::create(builder, loc, rowSpan, tids.x);
  Value bxBy = arith::MulIOp::create(builder, loc, sizes.x, sizes.y);
  Value wgSize = arith::MulIOp::create(builder, loc, bxBy, sizes.z);
  return std::pair<Value, Value>{linearTid, wgSize};
}
