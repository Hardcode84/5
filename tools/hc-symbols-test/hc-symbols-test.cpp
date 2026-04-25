// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCSymbols.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir::hc;

static void printLiteral(llvm::StringRef label, std::optional<int64_t> value) {
  llvm::outs() << label << ": ";
  if (value)
    llvm::outs() << *value;
  else
    llvm::outs() << "none";
  llvm::outs() << "\n";
}

int main() {
  sym::Store store;
  sym::Session session(store);

  printLiteral("int", sym::getIntegerLiteralValue(
                          sym::ExprHandle(ixs_int(session.raw(), -42))));
  printLiteral("unit-rat", sym::getIntegerLiteralValue(
                               sym::ExprHandle(ixs_rat(session.raw(), 7, 1))));
  printLiteral("non-unit-rat", sym::getIntegerLiteralValue(sym::ExprHandle(
                                   ixs_rat(session.raw(), 7, 2))));
  printLiteral("symbol", sym::getIntegerLiteralValue(
                             sym::ExprHandle(ixs_sym(session.raw(), "N"))));

  return 0;
}
