// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pass-internal helpers shared between the launch-body lowering passes.
// Anchored at `gpu.launch` boundary work — anything that maps the dim3
// thread / block layout to a linearised single-dim view of the wave
// belongs here.

#ifndef HC_TRANSFORMS_LAUNCH_UTILS_H
#define HC_TRANSFORMS_LAUNCH_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <utility>

namespace mlir::hc {

// Linearize the enclosing `gpu.launch`'s 3-D thread id and block size
// into a single `(tid, wgSize)` pair so cooperative-walk loops over a
// flat tile only have to reason about one dimension. The dim3 layout
// is the standard `lin = (tz * by + ty) * bx + tx`. For the common
// `block_y = block_z = 1` shape the y/z multiplications fold to
// identity and the resulting IR collapses to plain `tx` / `bx`.
//
// Returns failure when `anchor` has no enclosing `gpu.launch`; callers
// should treat that as a hard mismatch (a collective-dispatch shape
// outside a launch is ill-formed).
mlir::FailureOr<std::pair<mlir::Value, mlir::Value>>
linearizedThreadAndSize(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Operation *anchor);

} // namespace mlir::hc

#endif // HC_TRANSFORMS_LAUNCH_UTILS_H
