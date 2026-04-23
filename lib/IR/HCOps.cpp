// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"

using namespace mlir::hc;

#define GET_OP_CLASSES
#include "hc/IR/HCOps.cpp.inc"
