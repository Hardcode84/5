// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-bridge-intrinsics`: retype `hc.intrinsic` signatures and
// `hc.call_intrinsic` operand / result boundaries to upstream
// types via the same launch-body type converter, ahead of
// `hc-lower-launch-body`. Owns the conversion so launch-body sees
// intrinsics already in their final boundary shape.

#include "hc/Transforms/Passes.h"

#include "HCLowerLaunchBodyShared.h"

#include "hc/IR/HCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCBRIDGEINTRINSICS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct HCBridgeIntrinsicsPass
    : public hc::impl::HCBridgeIntrinsicsBase<HCBridgeIntrinsicsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> converter = makeLaunchBodyTypeConverter();
    RewritePatternSet patterns(ctx);
    populateIntrinsicBridgingPatterns(*converter, patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, HCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    registerIntrinsicBridgingLegality(*converter, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
