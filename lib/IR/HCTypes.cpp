// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCTypes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <string>

using namespace mlir;
using namespace mlir::hc;

namespace {

// `!hc.idx` and `!hc.pred` inline their symbolic text as a quoted string.
// Parsing routes through the dialect-owned ixsimpl symbolic expression store;
// printing reuses the canonical ixsimpl-rendered form so the handle-backed
// attribute stays the single source of truth.

template <typename HandleT>
using StoreParser = FailureOr<HandleT> (*)(sym::Store &, llvm::StringRef,
                                           std::string *);

// Accepts either the inline form (`"expr"`, the canonical printer output) or
// the explicit attribute form (`#hc.expr<"expr">`). `parseOptionalString`
// returns `success` iff the next token is a string literal, so we can branch
// on it without lookahead hacks.
template <typename AttrT, typename HandleT, StoreParser<HandleT> ParseFn>
FailureOr<AttrT> parseInlineOrAttrForm(AsmParser &parser,
                                       llvm::StringRef what) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::string text;
  if (succeeded(parser.parseOptionalString(&text))) {
    std::string diagnostic;
    auto *dialect = parser.getContext()->getOrLoadDialect<HCDialect>();
    FailureOr<HandleT> handle =
        ParseFn(dialect->getSymbolStore(), text, &diagnostic);
    if (failed(handle)) {
      parser.emitError(loc, diagnostic.empty()
                                ? llvm::Twine("invalid ") + what + " text"
                                : llvm::Twine(diagnostic));
      return failure();
    }
    return AttrT::get(parser.getContext(), *handle);
  }
  Attribute raw;
  if (parser.parseAttribute(raw))
    return failure();
  AttrT attr = llvm::dyn_cast<AttrT>(raw);
  if (!attr) {
    parser.emitError(loc) << "expected " << what << " attribute or inline text";
    return failure();
  }
  return attr;
}

void printInlineNode(AsmPrinter &printer, MLIRContext *ctx,
                     const ixs_node *node) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  printer.printString(store.render(node));
}

} // namespace

#define GET_TYPEDEF_CLASSES
#include "hc/IR/HCTypes.cpp.inc"

#include "hc/IR/HCTypesInterfaces.cpp.inc"

void HCDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/IR/HCTypes.cpp.inc"
      >();
}

IdxType mlir::hc::getUnpinnedIdxType(MLIRContext *ctx) {
  return IdxType::get(ctx, ExprAttr{});
}

PredType mlir::hc::getUnpinnedPredType(MLIRContext *ctx) {
  return PredType::get(ctx, PredAttr{});
}

mlir::LogicalResult
mlir::hc::BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

mlir::LogicalResult
mlir::hc::TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

mlir::LogicalResult
mlir::hc::VectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

Type IdxType::parse(AsmParser &parser) {
  if (failed(parser.parseOptionalLess()))
    return getUnpinnedIdxType(parser.getContext());

  FailureOr<ExprAttr> expr =
      parseInlineOrAttrForm<ExprAttr, sym::ExprHandle, sym::parseExpr>(
          parser, "hc.expr");
  if (failed(expr) || parser.parseGreater())
    return {};
  return IdxType::get(parser.getContext(), *expr);
}

void IdxType::print(AsmPrinter &printer) const {
  ExprAttr expr = getExpr();
  if (!expr)
    return;
  printer << "<";
  printInlineNode(printer, getContext(), expr.getNode());
  printer << ">";
}

Type IdxType::joinHCType(Type other) const {
  // Distinct symbolic facts widen to the unpinned idx type. Keeping a concrete
  // expression here would guess which control-flow predecessor won.
  if (isa<IdxType>(other))
    return getUnpinnedIdxType(getContext());
  return {};
}

Type PredType::parse(AsmParser &parser) {
  if (failed(parser.parseOptionalLess()))
    return getUnpinnedPredType(parser.getContext());

  FailureOr<PredAttr> pred =
      parseInlineOrAttrForm<PredAttr, sym::PredHandle, sym::parsePred>(
          parser, "hc.pred");
  if (failed(pred) || parser.parseGreater())
    return {};
  return PredType::get(parser.getContext(), *pred);
}

void PredType::print(AsmPrinter &printer) const {
  PredAttr pred = getPred();
  if (!pred)
    return;
  printer << "<";
  printInlineNode(printer, getContext(), pred.getNode());
  printer << ">";
}

Type PredType::joinHCType(Type other) const {
  // Predicates use the same conservative widening as idx expressions: preserve
  // the kind, drop the path-specific symbolic payload.
  if (isa<PredType>(other))
    return getUnpinnedPredType(getContext());
  return {};
}
