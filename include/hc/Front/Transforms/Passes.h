// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_TRANSFORMS_PASSES_H
#define HC_FRONT_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::hc::front {

#define GEN_PASS_DECL
#include "hc/Front/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hc/Front/Transforms/Passes.h.inc"

} // namespace mlir::hc::front

#endif // HC_FRONT_TRANSFORMS_PASSES_H
