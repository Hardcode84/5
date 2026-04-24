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

// Registers hc's three pass families (hc-front transforms, hc-front→hc
// conversion, hc transforms) into the process-wide pass registry.
//
// We deliberately do NOT call `mlir::registerAllPasses()` here: the Python
// bindings import `_mlirRegisterEverything` during site initialization,
// which already registers every upstream pass and pipeline. Re-registering
// pipelines (they use `PassPipelineRegistration` which throws on
// duplicates) would abort with "registered multiple times". Our three
// calls are safe because the upstream `registerHC*Passes` entry points
// check before inserting.
//
// `std::call_once` guards against callers invoking this once per context
// within the same process; the observable state — "hc pass names are
// resolvable by `PassManager::parse`" — is fully set after the first call.
void mlirRegisterHCAllPasses(void) {
  static std::once_flag flag;
  std::call_once(flag, []() {
    mlir::hc::front::registerHCFrontToHCConversionPasses();
    mlir::hc::front::registerHCFrontTransformsPasses();
    mlir::hc::registerHCTransformsPasses();
  });
}
