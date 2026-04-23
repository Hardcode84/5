// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCDIALECT
#define HC_IR_HCDIALECT

include "mlir/IR/OpBase.td"

def HC_Dialect : Dialect {
  let name = "hc";
  let summary = "Project hc semantic dialect.";
  let description = [{
    The `hc` MLIR dialect is the semantic kernel IR that `hc_front` lowers
    into. It owns symbolic attrs, semantic types, and region-bearing kernel
    operations.
  }];
  let cppNamespace = "::mlir::hc";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();

    sym::Store &getSymbolStore();
    const sym::Store &getSymbolStore() const;

  private:
    std::unique_ptr<sym::Store> symbolStore;
  }];
}

class HC_Op<string mnemonic, list<Trait> traits = []>
    : Op<HC_Dialect, mnemonic, traits>;

#endif // HC_IR_HCDIALECT
