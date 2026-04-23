// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES_H
#define HC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::hc {

// ODS-generated pass-base declarations plus free `createXxxPass()`
// factories. With no `let constructor` in `Passes.td`, tablegen emits
// the factory as a friend of the CRTP base so the DerivedT is the one
// we `std::make_unique<>` here.
#define GEN_PASS_DECL
#include "hc/Transforms/Passes.h.inc"

// `registerHCTransformsPasses()` — tablegen-generated; registers
// `-hc-promote-names` with the global `mlir-opt`-style registry.
#define GEN_PASS_REGISTRATION
#include "hc/Transforms/Passes.h.inc"

} // namespace mlir::hc

#endif // HC_TRANSFORMS_PASSES_H
