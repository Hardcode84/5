// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-apply`: per-lane apply rewriter. Runs after
// `hc-lower-generic` to consume the fresh `hc.idx_apply` /
// `hc.pred_apply` ops that body cloning planted outside the now-
// dissolved generic. Replaces the second `hc-lower-launch-body`
// invocation -- only the apply patterns + their legality need to fire
// at that stage, and re-spinning the full launch-body target wastes
// pass-startup work.

#include "hc/Transforms/Passes.h"

#include "HCLowerLaunchBodyShared.h"

#include "hc/IR/HCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERAPPLY
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct HCLowerApplyPass : public hc::impl::HCLowerApplyBase<HCLowerApplyPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> converter = makeLaunchBodyTypeConverter();
    RewritePatternSet patterns(ctx);
    populateLaunchBodyApplyPatterns(*converter, patterns, ctx);

    ConversionTarget target(*ctx);
    // Surrounding dialects stay legal; only the apply ops are
    // converted here. Unknown ops legal so apply-free IR is a no-op.
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           gpu::GPUDialect, HCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    registerLaunchBodyApplyDynamicLegality(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
