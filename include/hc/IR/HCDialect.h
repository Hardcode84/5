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
// Attach `DialectExtension`s anchored on `HCDialect` whose only job is to
// `getOrLoadDialect` upstream dialects that the canonical pipeline will
// eventually need. Today the list is just `dlti`: `transform-interpreter`
// dispatches `gpu-to-llvm` from within a multi-threaded PassManager, and
// `gpu-to-llvm` reaches for `dlti.dl_spec` on the module the first time it
// computes a pointer width. Lazy-loading `dlti` at that point trips
// MLIR's "Loading a dialect while in a multi-threaded execution context"
// guard. By piggy-backing on the synchronous, single-threaded load of
// `HCDialect` that the PassManager does up-front for its `dependentDialects`
// scan, we get `dlti` into the context before threading kicks in.
void registerHCDependentDialectExtensions(DialectRegistry &registry);
} // namespace hc
} // namespace mlir

#endif // HC_IR_HCDIALECT_H
