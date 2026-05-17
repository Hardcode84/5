// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_IR_HCFRONTDIALECT
#define HC_FRONT_IR_HCFRONTDIALECT

include "mlir/IR/OpBase.td"

def HCFront_Dialect : Dialect {
  let name = "hc_front";
  let summary = "Project hc frontend dialect for source-faithful AST serialization.";
  let description = [{
    Source-faithful frontend IR. Serializes the supported Python AST
    subset before semantic legalization into the `hc` dialect.

    Textual form: ops like `hc_front.kernel`, types like
    `!hc_front.value`. C++ API in `mlir::hc::front`.
  }];
  let cppNamespace = "::mlir::hc::front";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class HCFront_Op<string mnemonic, list<Trait> traits = []>
    : Op<HCFront_Dialect, mnemonic, traits>;

#endif // HC_FRONT_IR_HCFRONTDIALECT
