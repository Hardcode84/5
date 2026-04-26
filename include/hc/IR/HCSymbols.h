// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_IR_HCSYMBOLS_H
#define HC_IR_HCSYMBOLS_H

#include "ixsimpl.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Mutex.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir

namespace mlir::hc::sym {

/// Hash-consed symbolic expression handle scoped to one Store/ixs_ctx.
struct ExprHandle {
  ExprHandle() = default;
  explicit ExprHandle(const ixs_node *node) : node(node) {}

  const ixs_node *raw() const { return node; }
  explicit operator bool() const { return node != nullptr; }

  friend bool operator==(ExprHandle lhs, ExprHandle rhs) {
    return lhs.node == rhs.node;
  }
  friend llvm::hash_code hash_value(ExprHandle value) {
    return llvm::hash_value(value.node);
  }

private:
  const ixs_node *node = nullptr;
};

/// Hash-consed symbolic predicate handle scoped to one Store/ixs_ctx.
struct PredHandle {
  PredHandle() = default;
  explicit PredHandle(const ixs_node *node) : node(node) {}

  const ixs_node *raw() const { return node; }
  explicit operator bool() const { return node != nullptr; }

  friend bool operator==(PredHandle lhs, PredHandle rhs) {
    return lhs.node == rhs.node;
  }
  friend llvm::hash_code hash_value(PredHandle value) {
    return llvm::hash_value(value.node);
  }

private:
  const ixs_node *node = nullptr;
};

enum class ExprBinaryOp { Add, Sub, Mul, Div, Mod };
enum class PredCmpOp { Lt, Le, Gt, Ge, Eq, Ne };

/// Long-lived symbolic store owned by one `HCDialect` / `MLIRContext`.
class Store {
public:
  Store();
  ~Store();

  Store(const Store &) = delete;
  Store &operator=(const Store &) = delete;

  ixs_ctx *raw() const { return ctx; }
  /// Renders through the store while holding the store mutex. Use this from
  /// store-centric helpers that already traffic in `Store &`.
  std::string render(const ixs_node *node) const;

private:
  ixs_ctx *ctx = nullptr;
  mutable llvm::sys::SmartMutex<true> mutex;

  friend class Session;
};

class Session {
public:
  explicit Session(Store &store);
  ~Session();

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  ixs_session *raw() { return &session; }

private:
  Store &store;
  llvm::sys::SmartScopedLock<true> lock;
  ixs_session session;
};

/// Parses symbolic text into the destination store. The helper copies the
/// input to satisfy the upstream NUL-terminated parser contract.
mlir::FailureOr<ExprHandle> parseExpr(Store &store, llvm::StringRef text,
                                      std::string *diagnostic = nullptr);
mlir::FailureOr<PredHandle> parsePred(Store &store, llvm::StringRef text,
                                      std::string *diagnostic = nullptr);

mlir::FailureOr<ExprHandle> importExpr(Store &store, const ixs_node *foreign,
                                       std::string *diagnostic = nullptr);
mlir::FailureOr<PredHandle> importPred(Store &store, const ixs_node *foreign,
                                       std::string *diagnostic = nullptr);

mlir::FailureOr<ExprHandle>
composeExprBinary(Store &store, ExprHandle lhs, ExprBinaryOp op, ExprHandle rhs,
                  std::string *diagnostic = nullptr);
mlir::FailureOr<ExprHandle> composeExprCeil(Store &store, ExprHandle value,
                                            std::string *diagnostic = nullptr);
mlir::FailureOr<ExprHandle> composeExprNeg(Store &store, ExprHandle value,
                                           std::string *diagnostic = nullptr);
mlir::FailureOr<PredHandle> composePredCmp(Store &store, ExprHandle lhs,
                                           PredCmpOp op, ExprHandle rhs,
                                           std::string *diagnostic = nullptr);

/// Returns the integer payload when an expression is structurally integral:
/// an ixs integer node, or a rational node with unit denominator. Does not
/// render or parse the expression text.
std::optional<int64_t> getIntegerLiteralValue(ExprHandle value);

/// Walk every symbolic leaf name in a hash-consed expression or predicate.
/// Traversal is structural and does not render or parse the expression text.
void walkSymbolNames(ExprHandle value,
                     llvm::function_ref<void(llvm::StringRef)> callback);
void walkSymbolNames(PredHandle value,
                     llvm::function_ref<void(llvm::StringRef)> callback);

mlir::FailureOr<ExprHandle> parseExprHandle(AsmParser &parser);
mlir::FailureOr<PredHandle> parsePredHandle(AsmParser &parser);

/// ODS printer hooks only receive `AsmPrinter` plus the handle value. They
/// intentionally render directly from the immutable hash-consed node instead of
/// reacquiring the store mutex through the dialect.
void printExprHandle(AsmPrinter &printer, ExprHandle value);
void printPredHandle(AsmPrinter &printer, PredHandle value);

} // namespace mlir::hc::sym

#endif // HC_IR_HCSYMBOLS_H
