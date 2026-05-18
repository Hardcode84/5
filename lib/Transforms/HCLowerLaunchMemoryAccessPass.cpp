// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-memory-access`: load / store / buffer_view /
// vec / select slice of the old `hc-lower-launch-body` rewrites.
// Walks the kernel-arg UCC bundle to resolve `!hc.buffer` operands,
// emits per-element `hc.ptr_offset` + `hc.ptr_load[_pred]` /
// `hc.ptr_store[_pred]` + `vector` ops through the launch-body type
// converter.

#include "hc/Transforms/Passes.h"

#include "HCLowerLaunchBodyShared.h"

#include "hc/IR/HCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLAUNCHMEMORYACCESS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct HCLowerLaunchMemoryAccessPass
    : public hc::impl::HCLowerLaunchMemoryAccessBase<
          HCLowerLaunchMemoryAccessPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> converter = makeLaunchBodyTypeConverter();
    RewritePatternSet patterns(ctx);
    populateLaunchMemoryAccessPatterns(*converter, patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect,
                           HCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    registerLaunchMemoryAccessLegality(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
