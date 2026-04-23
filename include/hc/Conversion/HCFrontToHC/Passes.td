// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_CONVERSION_HCFRONTTOHC_PASSES
#define HC_CONVERSION_HCFRONTTOHC_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHCFrontToHC : Pass<"convert-hc-front-to-hc", "::mlir::ModuleOp"> {
  let summary = "Mechanical hc_front -> hc lowering driven by `ref` metadata.";
  let description = [{
    Walks every `hc_front.kernel` / `hc_front.func` / `hc_front.intrinsic`
    top-level op and emits the parallel `hc` callable, consuming the `ref`
    DictAttr stamped by the Python driver on each `hc_front.name` /
    `hc_front.attr`. Every produced value has `!hc.undef`; a later inference
    pass pins concrete types.
  }];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::hc::front::HCFrontDialect",
  ];
}

#endif // HC_CONVERSION_HCFRONTTOHC_PASSES
