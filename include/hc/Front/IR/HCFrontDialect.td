// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_FRONT_IR_HCFRONTDIALECT
#define HC_FRONT_IR_HCFRONTDIALECT

include "mlir/IR/OpBase.td"

def HCFront_Dialect : Dialect {
  let name = "hc";
  let summary = "Project hc dialect hosting the source-faithful front.* IR family.";
  let description = [{
    The `hc` MLIR dialect currently hosts the `front.*` source-faithful IR
    family used to serialize the supported Python AST subset. Textual frontend
    IR therefore appears as operations such as `hc.front.kernel` and types such
    as `!hc.front.value`, even though the registered dialect namespace reported
    by MLIR tooling is `hc`.

    This bootstrap layout keeps the frontend surface low-intelligence and
    source-preserving while leaving room for later semantic hc operations to
    live under the same top-level project dialect namespace.
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
