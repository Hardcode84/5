// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Internal pass-side header: populators exposed by
// `HCLowerLaunchBodyPass.cpp` for the post-`hc-lower-generic`
// `hc-lower-apply` pass to reuse the apply lowering patterns.

#ifndef HC_TRANSFORMS_HC_LOWER_LAUNCH_BODY_SHARED_H
#define HC_TRANSFORMS_HC_LOWER_LAUNCH_BODY_SHARED_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir::hc {

// Same converter `hc-lower-launch-body` uses: `!hc.idx` -> `index`,
// `!hc.pred` -> `i1`, bare carriers -> `vector` / `!hc.ptr<workgroup>`,
// tuples recurse, source/target materialisations emit
// `unrealized_conversion_cast`. Returned by value-via-pointer because
// the converter type is TU-local to launch-body.
std::unique_ptr<TypeConverter> makeLaunchBodyTypeConverter();

// Populate `hc.idx_apply` + `hc.pred_apply` conversion patterns
// against `converter`. Same `ExprLowerer` plumbing as the launch-body
// pass.
void populateLaunchBodyApplyPatterns(TypeConverter &converter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx);

// Mark applies / `hc.predicate` dynamic-legal inside `hc.generic`
// body, illegal everywhere else.
void registerLaunchBodyApplyDynamicLegality(ConversionTarget &target);

// Populate `hc.intrinsic` + `hc.call_intrinsic` boundary type
// conversion patterns. Same boundary type rules as `hc-lower-launch-body`.
void populateIntrinsicBridgingPatterns(TypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       MLIRContext *ctx);

// Mark `hc.intrinsic` / `hc.call_intrinsic` dyn-legal once their
// signatures / boundaries match `converter`'s projection.
void registerIntrinsicBridgingLegality(const TypeConverter &converter,
                                       ConversionTarget &target);

// Populate the `hc.generic` operand reconciliation pattern.
// Resolves kernel-arg `!hc.buffer` -> `!hc.ptr<global>` and LDS
// `!hc.bare_tensor` ins -> `!hc.ptr<workgroup>`.
void populateGenericReconciliationPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           MLIRContext *ctx);

// `hc.generic` is dyn-legal once its ins / outs no longer carry
// unresolved kernel-arg buffers or LDS bare_tensor sources.
void registerGenericReconciliationLegality(ConversionTarget &target);

// Populate scalar / control-flow HC ops (const, int/float arith,
// cmp, cast, for_range, if). Each pattern lowers to its arith /
// scf counterpart through `converter`.
void populateLaunchScalarOpsPatterns(TypeConverter &converter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx);

// Mark every scalar / control-flow HC op illegal so `populate`'s
// patterns are the only path out.
void registerLaunchScalarOpsLegality(ConversionTarget &target);

// Populate the shaped-constant family (`hc.zeros` / `hc.ones` /
// `hc.full` / their `v` vector variants / `hc.empty` / `hc.full_mask`).
// LDS-result variants plant `hc.alloc workgroup` + `hc.generic` fills;
// vector-result variants stay as `arith.constant` splats.
void populateLaunchShapedConstantsPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           MLIRContext *ctx);

// Mark every shaped-constant op illegal so the patterns above are
// the only lowering path.
void registerLaunchShapedConstantsLegality(ConversionTarget &target);

void populateLaunchMemoryAccessPatterns(TypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void registerLaunchMemoryAccessLegality(ConversionTarget &target);

} // namespace mlir::hc

#endif // HC_TRANSFORMS_HC_LOWER_LAUNCH_BODY_SHARED_H
