// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES_H
#define HC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::hc {

// -hc-promote-names: rebuilds the IR so every `hc.assign` / `hc.name_load`
// becomes a real SSA edge, threading region-carrying ops with iter_args /
// results / yields as needed. See `Passes.td` for the full contract.
std::unique_ptr<Pass> createPromoteNamesPass();

// Forward declarations for the ODS-generated pass bases.
#define GEN_PASS_DECL
#include "hc/Transforms/Passes.h.inc"

// `registerHCTransformsPasses()` — tablegen-generated; registers
// `-hc-promote-names` with the global `mlir-opt`-style registry.
#define GEN_PASS_REGISTRATION
#include "hc/Transforms/Passes.h.inc"

} // namespace mlir::hc

#endif // HC_TRANSFORMS_PASSES_H
