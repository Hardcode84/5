// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-c/HC.h"

#include "hc/Conversion/HCFrontToHC/HCFrontToHC.h"
#include "hc/Front/Transforms/Passes.h"
#include "hc/IR/HCDialect.h"
#include "hc/Transforms/Passes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

#include <mutex>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HC, hc, mlir::hc::HCDialect)

// See HC.h for the contract around upstream pass registration
// (`registerAllPasses` is intentionally not called here). `std::call_once`
// guards the idempotency the header promises.
void mlirRegisterHCAllPasses(void) {
  static std::once_flag flag;
  std::call_once(flag, []() {
    mlir::hc::front::registerHCFrontToHCConversionPasses();
    mlir::hc::front::registerHCFrontTransformsPasses();
    mlir::hc::registerHCTransformsPasses();
  });
}
