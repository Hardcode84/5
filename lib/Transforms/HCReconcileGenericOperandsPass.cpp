// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-reconcile-generic-operands`: post-launch-body cleanup that
// resolves `hc.generic` ins / outs to upstream-compatible carriers.
// Kernel-arg `!hc.buffer` bundles walk back through their UCC chains
// to the underlying `!hc.ptr<global>`; LDS-backed `!hc.bare_tensor`
// ins swap to the producer-side `!hc.ptr<workgroup>` shaped-constant
// lowering planted. Outs stay bare_tensor (the value-typed-outs SSA
// result parity verifier reads off that type).

#include "hc/Transforms/Passes.h"

#include "HCLowerLaunchBodyShared.h"

#include "hc/IR/HCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCRECONCILEGENERICOPERANDS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

struct HCReconcileGenericOperandsPass
    : public hc::impl::HCReconcileGenericOperandsBase<
          HCReconcileGenericOperandsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::unique_ptr<TypeConverter> converter = makeLaunchBodyTypeConverter();
    RewritePatternSet patterns(ctx);
    populateGenericReconciliationPatterns(*converter, patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, HCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    registerGenericReconciliationLegality(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
