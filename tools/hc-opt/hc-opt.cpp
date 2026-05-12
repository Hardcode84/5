// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Conversion/HCFrontToHC/HCFrontToHC.h"
#include "hc/Front/IR/HCFrontDialect.h"
#include "hc/Front/Transforms/Passes.h"
#include "hc/IR/HCDialect.h"
#include "hc/TransformOps/HCTransformOps.h"
#include "hc/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  // Register LLVM IR translation interfaces so passes that reach for
  // translateModuleToLLVMIR (notably gpu-module-to-binary on the rocdl path)
  // can find the per-dialect translation hooks. Without this the GPU binary
  // emission pass fails at the IR-translation step before it ever shells out
  // to lld.
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerAllPasses();
  mlir::hc::front::registerHCFrontToHCConversionPasses();
  mlir::hc::front::registerHCFrontTransformsPasses();
  mlir::hc::registerHCTransformsPasses();
  mlir::hc::transform::registerTransformDialectExtension(registry);
  registry.insert<mlir::hc::HCDialect>();
  registry.insert<mlir::hc::front::HCFrontDialect>();
  // Pre-load `dlti` (etc.) the moment HC is loaded, so passes that get
  // dispatched from inside `transform-interpreter`'s threaded inner PM
  // don't race against MLIR's "Loading a dialect (dlti) while in a
  // multi-threaded execution context" guard. The PassManager loads
  // dependent dialects synchronously before threading kicks in; HC is a
  // dependent dialect of every meaningful pipeline, so its extension is
  // the right hook.
  mlir::hc::registerHCDependentDialectExtensions(registry);

  auto result = mlir::MlirOptMain(
      argc, argv, "hc optimizer driver with hc and hc_front dialect support\n",
      registry);
  return mlir::failed(result) ? 1 : 0;
}
