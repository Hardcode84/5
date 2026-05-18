// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-shaped-constants`: materialise shaped constants
// (`hc.zeros` / `hc.ones` / `hc.full` / vector variants / `hc.empty`
// / `hc.full_mask`) ahead of the main launch-body lowering. LDS-
// result variants emit `hc.alloc workgroup` + `hc.generic` fills;
// vector-result variants stay as `arith.constant` splats.

#include "hc/Transforms/Passes.h"

#include "HCLowerLaunchBodyShared.h"

#include "hc/IR/HCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLAUNCHSHAPEDCONSTANTS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct HCLowerLaunchShapedConstantsPass
    : public hc::impl::HCLowerLaunchShapedConstantsBase<
          HCLowerLaunchShapedConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> converter = makeLaunchBodyTypeConverter();
    RewritePatternSet patterns(ctx);
    populateLaunchShapedConstantsPatterns(*converter, patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect,
                           HCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    registerLaunchShapedConstantsLegality(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
