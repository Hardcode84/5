// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCSymbols.h"

#include "hc/IR/HCDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::hc;
using namespace mlir::hc::sym;

namespace {

std::string joinSessionErrors(ixs_session *session) {
  std::string message;
  llvm::raw_string_ostream os(message);
  size_t nerrors = ixs_session_nerrors(session);
  for (size_t i = 0; i < nerrors; ++i) {
    if (i)
      os << "; ";
    os << ixs_session_error(session, i);
  }
  return message;
}

std::string renderNode(const ixs_node *node) {
  assert(node && "expected non-null ixsimpl node");
  // The vendored ixsimpl printer is a pure read-only walk over the immutable
  // node graph, so rendering does not need an ixs_session scratch object.
  size_t n = ixs_print(const_cast<ixs_node *>(node), nullptr, 0);
  if (n == std::numeric_limits<size_t>::max())
    llvm::report_fatal_error("hc symbolic printer reported an invalid length");
  std::string text(n + 1, '\0');
  ixs_print(const_cast<ixs_node *>(node), text.data(), text.size());
  text.resize(n);
  return text;
}

HCDialect &getHCDialect(MLIRContext *context) {
  return *context->getOrLoadDialect<HCDialect>();
}

void setDiagnostic(std::string *diagnostic, std::string message) {
  if (diagnostic)
    *diagnostic = std::move(message);
}

ixs_node *importNode(Session &session, const ixs_node *node,
                     std::string *diagnostic, const char *kind) {
  if (!node) {
    setDiagnostic(diagnostic, std::string("cannot compose null ") + kind);
    return nullptr;
  }
  ixs_node *imported = ixs_import_node(session.raw(), node);
  if (!imported)
    setDiagnostic(diagnostic, std::string("out of memory importing ") + kind);
  return imported;
}

FailureOr<ExprHandle> finishExpr(ixs_session *session, ixs_node *node,
                                 std::string *diagnostic,
                                 const char *fallback) {
  if (!node) {
    setDiagnostic(diagnostic, fallback);
    return failure();
  }
  if (!ixs_node_is_expr(node)) {
    std::string message = joinSessionErrors(session);
    setDiagnostic(diagnostic, message.empty() ? fallback : message);
    return failure();
  }
  return ExprHandle(node);
}

FailureOr<PredHandle> finishPred(ixs_session *session, ixs_node *node,
                                 std::string *diagnostic,
                                 const char *fallback) {
  if (!node) {
    setDiagnostic(diagnostic, fallback);
    return failure();
  }
  if (!ixs_node_is_pred(node)) {
    std::string message = joinSessionErrors(session);
    setDiagnostic(diagnostic, message.empty() ? fallback : message);
    return failure();
  }
  return PredHandle(node);
}

} // namespace

Store::Store() : ctx(ixs_ctx_create()) {
  if (!ctx)
    llvm::report_fatal_error("failed to create hc symbolic ixsimpl store");
}

Store::~Store() { ixs_ctx_destroy(ctx); }

std::string Store::render(const ixs_node *node) const {
  // Keep a serialized rendering entry point for store-centric callers that want
  // one obvious synchronization policy around the dialect-owned context.
  llvm::sys::SmartScopedLock<true> lock(mutex);
  return renderNode(node);
}

Session::Session(Store &store) : store(store), lock(store.mutex) {
  ixs_session_init(&session, store.ctx);
}

Session::~Session() { ixs_session_destroy(&session); }

FailureOr<ExprHandle> mlir::hc::sym::parseExpr(Store &store,
                                               llvm::StringRef text,
                                               std::string *diagnostic) {
  std::string nulTerminated(text);
  Session session(store);
  ixs_node *node = ixs_parse_expr(session.raw(), nulTerminated.c_str(),
                                  nulTerminated.size());
  if (!node) {
    setDiagnostic(diagnostic, "out of memory parsing hc.expr");
    return failure();
  }
  if (!ixs_node_is_expr(node)) {
    std::string message = joinSessionErrors(session.raw());
    setDiagnostic(diagnostic, message.empty()
                                  ? "invalid hc.expr text"
                                  : "invalid hc.expr text: " + message);
    return failure();
  }
  return ExprHandle(node);
}

FailureOr<PredHandle> mlir::hc::sym::parsePred(Store &store,
                                               llvm::StringRef text,
                                               std::string *diagnostic) {
  std::string nulTerminated(text);
  Session session(store);
  ixs_node *node = ixs_parse_pred(session.raw(), nulTerminated.c_str(),
                                  nulTerminated.size());
  if (!node) {
    setDiagnostic(diagnostic, "out of memory parsing hc.pred");
    return failure();
  }
  if (!ixs_node_is_pred(node)) {
    std::string message = joinSessionErrors(session.raw());
    setDiagnostic(diagnostic, message.empty()
                                  ? "invalid hc.pred text"
                                  : "invalid hc.pred text: " + message);
    return failure();
  }
  return PredHandle(node);
}

FailureOr<ExprHandle> mlir::hc::sym::importExpr(Store &store,
                                                const ixs_node *foreign,
                                                std::string *diagnostic) {
  if (!foreign) {
    setDiagnostic(diagnostic, "cannot import null hc.expr node");
    return failure();
  }
  if (!ixs_node_is_expr(foreign)) {
    setDiagnostic(diagnostic, "expected expression node for hc.expr");
    return failure();
  }

  Session session(store);
  ixs_node *node = ixs_import_node(session.raw(), foreign);
  if (!node) {
    setDiagnostic(diagnostic, "out of memory importing hc.expr");
    return failure();
  }
  return ExprHandle(node);
}

FailureOr<PredHandle> mlir::hc::sym::importPred(Store &store,
                                                const ixs_node *foreign,
                                                std::string *diagnostic) {
  if (!foreign) {
    setDiagnostic(diagnostic, "cannot import null hc.pred node");
    return failure();
  }
  if (!ixs_node_is_pred(foreign)) {
    setDiagnostic(diagnostic, "expected predicate node for hc.pred");
    return failure();
  }

  Session session(store);
  ixs_node *node = ixs_import_node(session.raw(), foreign);
  if (!node) {
    setDiagnostic(diagnostic, "out of memory importing hc.pred");
    return failure();
  }
  return PredHandle(node);
}

FailureOr<ExprHandle>
mlir::hc::sym::composeExprBinary(Store &store, ExprHandle lhsHandle,
                                 ExprBinaryOp op, ExprHandle rhsHandle,
                                 std::string *diagnostic) {
  Session session(store);
  ixs_node *lhs = importNode(session, lhsHandle.raw(), diagnostic, "hc.expr");
  ixs_node *rhs = importNode(session, rhsHandle.raw(), diagnostic, "hc.expr");
  if (!lhs || !rhs)
    return failure();

  ixs_node *node = nullptr;
  switch (op) {
  case ExprBinaryOp::Add:
    node = ixs_add(session.raw(), lhs, rhs);
    break;
  case ExprBinaryOp::Sub:
    node = ixs_sub(session.raw(), lhs, rhs);
    break;
  case ExprBinaryOp::Mul:
    node = ixs_mul(session.raw(), lhs, rhs);
    break;
  case ExprBinaryOp::Div:
    node = ixs_div(session.raw(), lhs, rhs);
    break;
  case ExprBinaryOp::Mod:
    node = ixs_mod(session.raw(), lhs, rhs);
    break;
  }
  return finishExpr(session.raw(), node, diagnostic,
                    "failed to compose hc.expr");
}

FailureOr<ExprHandle> mlir::hc::sym::composeExprNeg(Store &store,
                                                    ExprHandle valueHandle,
                                                    std::string *diagnostic) {
  Session session(store);
  ixs_node *value =
      importNode(session, valueHandle.raw(), diagnostic, "hc.expr");
  if (!value)
    return failure();
  return finishExpr(session.raw(), ixs_neg(session.raw(), value), diagnostic,
                    "failed to compose hc.expr");
}

FailureOr<PredHandle>
mlir::hc::sym::composePredCmp(Store &store, ExprHandle lhsHandle, PredCmpOp op,
                              ExprHandle rhsHandle, std::string *diagnostic) {
  Session session(store);
  ixs_node *lhs = importNode(session, lhsHandle.raw(), diagnostic, "hc.expr");
  ixs_node *rhs = importNode(session, rhsHandle.raw(), diagnostic, "hc.expr");
  if (!lhs || !rhs)
    return failure();

  ixs_cmp_op cmp = IXS_CMP_EQ;
  switch (op) {
  case PredCmpOp::Lt:
    cmp = IXS_CMP_LT;
    break;
  case PredCmpOp::Le:
    cmp = IXS_CMP_LE;
    break;
  case PredCmpOp::Gt:
    cmp = IXS_CMP_GT;
    break;
  case PredCmpOp::Ge:
    cmp = IXS_CMP_GE;
    break;
  case PredCmpOp::Eq:
    cmp = IXS_CMP_EQ;
    break;
  case PredCmpOp::Ne:
    cmp = IXS_CMP_NE;
    break;
  }
  return finishPred(session.raw(), ixs_cmp(session.raw(), lhs, cmp, rhs),
                    diagnostic, "failed to compose hc.pred");
}

FailureOr<ExprHandle> mlir::hc::sym::parseExprHandle(AsmParser &parser) {
  std::string text;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseString(&text))
    return failure();

  std::string diagnostic;
  FailureOr<ExprHandle> handle = parseExpr(
      getHCDialect(parser.getContext()).getSymbolStore(), text, &diagnostic);
  if (failed(handle)) {
    parser.emitError(loc,
                     diagnostic.empty() ? "invalid hc.expr text" : diagnostic);
    return failure();
  }
  return *handle;
}

FailureOr<PredHandle> mlir::hc::sym::parsePredHandle(AsmParser &parser) {
  std::string text;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseString(&text))
    return failure();

  std::string diagnostic;
  FailureOr<PredHandle> handle = parsePred(
      getHCDialect(parser.getContext()).getSymbolStore(), text, &diagnostic);
  if (failed(handle)) {
    parser.emitError(loc,
                     diagnostic.empty() ? "invalid hc.pred text" : diagnostic);
    return failure();
  }
  return *handle;
}

void mlir::hc::sym::printExprHandle(AsmPrinter &printer, ExprHandle value) {
  // ODS printer hooks do not receive Store&; render directly from the immutable
  // node instead of re-looking up the dialect just to take the store lock.
  printer.printString(renderNode(value.raw()));
}

void mlir::hc::sym::printPredHandle(AsmPrinter &printer, PredHandle value) {
  printer.printString(renderNode(value.raw()));
}
