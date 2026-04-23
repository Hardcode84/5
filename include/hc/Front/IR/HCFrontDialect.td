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
    The `hc_front` MLIR dialect is the source-faithful frontend IR used to
    serialize the supported Python AST subset before semantic legalization into
    the separate `hc` dialect.

    Textual frontend IR therefore appears as operations such as
    `hc_front.kernel` and types such as `!hc_front.value`. The generated C++
    API remains in the `mlir::hc::front` namespace.
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
