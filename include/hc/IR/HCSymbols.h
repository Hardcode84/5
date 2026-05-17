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
  /// Render under the store mutex. For helpers already holding a `Store &`.
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

/// Parses symbolic text into the destination store. Copies the input to
/// satisfy the upstream NUL-terminated parser contract.
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
/// `composeExprBinary` with `Div` is *exact rational*. For Python `//`
/// (floored integer division) wrap a Div in `composeExprFloor`.
mlir::FailureOr<ExprHandle> composeExprFloor(Store &store, ExprHandle value,
                                             std::string *diagnostic = nullptr);
mlir::FailureOr<ExprHandle> composeExprNeg(Store &store, ExprHandle value,
                                           std::string *diagnostic = nullptr);

/// Symbol / integer leaves. Go through the dialect store so structurally
/// built expressions share hash-consed nodes with parsed ones.
mlir::FailureOr<ExprHandle> composeExprSym(Store &store, llvm::StringRef name,
                                           std::string *diagnostic = nullptr);
mlir::FailureOr<ExprHandle> composeExprInt(Store &store, int64_t value,
                                           std::string *diagnostic = nullptr);
mlir::FailureOr<PredHandle> composePredCmp(Store &store, ExprHandle lhs,
                                           PredCmpOp op, ExprHandle rhs,
                                           std::string *diagnostic = nullptr);
/// Hash-consed AND / OR of two predicates.
mlir::FailureOr<PredHandle> composePredAnd(Store &store, PredHandle lhs,
                                           PredHandle rhs,
                                           std::string *diagnostic = nullptr);
mlir::FailureOr<PredHandle> composePredOr(Store &store, PredHandle lhs,
                                          PredHandle rhs,
                                          std::string *diagnostic = nullptr);

/// Integer payload of a structurally-integral expression (ixs integer node
/// or unit-denominator rational). Structural, no parse / render.
std::optional<int64_t> getIntegerLiteralValue(ExprHandle value);

/// Walk every symbolic leaf name. Structural, no parse / render.
void walkSymbolNames(ExprHandle value,
                     llvm::function_ref<void(llvm::StringRef)> callback);
void walkSymbolNames(PredHandle value,
                     llvm::function_ref<void(llvm::StringRef)> callback);

mlir::FailureOr<ExprHandle> parseExprHandle(AsmParser &parser);
mlir::FailureOr<PredHandle> parsePredHandle(AsmParser &parser);

/// Render straight from the immutable hash-consed node — these don't take
/// `Store &` and won't reacquire the store mutex through the dialect.
void printExprHandle(AsmPrinter &printer, ExprHandle value);
void printPredHandle(AsmPrinter &printer, PredHandle value);

} // namespace mlir::hc::sym

#endif // HC_IR_HCSYMBOLS_H
