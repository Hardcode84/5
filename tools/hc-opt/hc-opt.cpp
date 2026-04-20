// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Front/IR/HCFrontDialect.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  registry.insert<mlir::hc::front::HCFrontDialect>();

  auto result = mlir::MlirOptMain(
      argc, argv, "hc optimizer driver with hc.front dialect support\n",
      registry);
  return mlir::failed(result) ? 1 : 0;
}
