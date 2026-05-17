// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCDIALECT_H
#define HC_IR_HCDIALECT_H

#include "hc/IR/HCSymbols.h"
#include "mlir/IR/Dialect.h"

#include <memory>

#include "hc/IR/HCOpsDialect.h.inc"

namespace mlir {
class DialectRegistry;

namespace hc {
// Force-load `dlti` synchronously when HCDialect loads. Avoids the
// multi-threaded dialect-load trip when `gpu-to-llvm` (dispatched from
// `transform-interpreter`) queries `dlti.dl_spec`.
void registerHCDependentDialectExtensions(DialectRegistry &registry);
} // namespace hc
} // namespace mlir

#endif // HC_IR_HCDIALECT_H
