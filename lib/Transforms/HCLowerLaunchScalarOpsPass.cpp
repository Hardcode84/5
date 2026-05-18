// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-launch-scalar-ops`: arith / scf cleanup for the
// scalar / control-flow HC ops surviving inside `gpu.launch` after
// the higher-level rewriters ran. Same launch-body type converter
// so `!hc.idx` -> `index` / `!hc.pred` -> `i1` materialisations
// match what `hc-lower-launch-body` produces downstream.

#include "hc/Transforms/Passes.h"

#include "HCLowerLaunchBodyShared.h"

#include "hc/IR/HCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLAUNCHSCALAROPS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct HCLowerLaunchScalarOpsPass
    : public hc::impl::HCLowerLaunchScalarOpsBase<HCLowerLaunchScalarOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> converter = makeLaunchBodyTypeConverter();
    RewritePatternSet patterns(ctx);
    populateLaunchScalarOpsPatterns(*converter, patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, HCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    registerLaunchScalarOpsLegality(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
