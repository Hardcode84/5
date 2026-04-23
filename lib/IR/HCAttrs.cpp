// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCAttrs.h"

#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::hc;

#define GET_ATTRDEF_CLASSES
#include "hc/IR/HCAttrs.cpp.inc"

namespace {

ParseResult parseShapeDim(AsmParser &parser, SmallVectorImpl<Attribute> &dims) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::string text;
  OptionalParseResult parsedString = parser.parseOptionalString(&text);
  if (!parsedString.has_value())
    return parser.emitError(loc) << "expected quoted hc.shape dimension";
  if (failed(*parsedString))
    return failure();

  std::string diagnostic;
  auto *dialect = parser.getContext()->getOrLoadDialect<HCDialect>();
  FailureOr<sym::ExprHandle> handle =
      sym::parseExpr(dialect->getSymbolStore(), text, &diagnostic);
  if (failed(handle))
    return parser.emitError(loc, diagnostic.empty() ? "invalid hc.shape dim"
                                                    : diagnostic);
  dims.push_back(ExprAttr::get(parser.getContext(), *handle));
  return success();
}

FailureOr<ShapeAttr> parseShapeDims(AsmParser &parser,
                                    bool openingBracketConsumed) {
  SmallVector<Attribute> dims;
  if (openingBracketConsumed) {
    if (failed(parser.parseOptionalRSquare())) {
      if (failed(parseShapeDim(parser, dims)))
        return failure();
      while (succeeded(parser.parseOptionalComma())) {
        if (failed(parseShapeDim(parser, dims)))
          return failure();
      }
      if (parser.parseRSquare())
        return failure();
    }
  } else if (parser.parseCommaSeparatedList(
                 AsmParser::Delimiter::Square,
                 [&]() { return parseShapeDim(parser, dims); })) {
    return failure();
  }
  return ShapeAttr::get(parser.getContext(), dims);
}

void printShapeDims(AsmPrinter &printer, ShapeAttr shape) {
  auto &store =
      shape.getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
  printer << "[";
  llvm::interleaveComma(shape.getDims(), printer, [&](Attribute dim) {
    printer.printString(store.render(llvm::cast<ExprAttr>(dim).getNode()));
  });
  printer << "]";
}

} // namespace

void HCDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/IR/HCAttrs.cpp.inc"
      >();
}

FailureOr<ShapeAttr> mlir::hc::parseInlineShapeAttr(AsmParser &parser) {
  if (succeeded(parser.parseOptionalLSquare())) {
    FailureOr<ShapeAttr> shape =
        parseShapeDims(parser, /*openingBracketConsumed=*/true);
    if (failed(shape))
      return failure();
    return *shape;
  }

  llvm::SMLoc loc = parser.getCurrentLocation();
  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  ShapeAttr shape = llvm::dyn_cast<ShapeAttr>(attr);
  if (!shape) {
    parser.emitError(loc, "expected #hc.shape attribute");
    return failure();
  }
  return shape;
}

void mlir::hc::printInlineShapeAttr(AsmPrinter &printer, ShapeAttr attr) {
  printShapeDims(printer, attr);
}

LogicalResult ExprAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               sym::ExprHandle value) {
  if (!value || !ixs_node_is_expr(value.raw()))
    return emitError() << "expected expression handle";
  return success();
}

LogicalResult PredAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               sym::PredHandle value) {
  if (!value || !ixs_node_is_pred(value.raw()))
    return emitError() << "expected predicate handle";
  return success();
}

Attribute ShapeAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  FailureOr<ShapeAttr> shape =
      parseShapeDims(parser, /*openingBracketConsumed=*/false);
  if (failed(shape) || parser.parseGreater())
    return {};

  return *shape;
}

void ShapeAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printShapeDims(printer, *this);
  printer << ">";
}

LogicalResult ShapeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<Attribute> dims) {
  for (Attribute dim : dims) {
    if (!llvm::isa<ExprAttr>(dim))
      return emitError() << "expected shape dims to be #hc.expr attributes";
  }
  return success();
}

LogicalResult
ConstraintSetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<Attribute> predicates) {
  for (Attribute predicate : predicates) {
    if (!llvm::isa<PredAttr>(predicate))
      return emitError()
             << "expected constraints to contain only #hc.pred attributes";
  }
  return success();
}

LogicalResult ScopeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                StringRef name) {
  if (name != "WorkGroup" && name != "SubGroup" && name != "WorkItem")
    return emitError() << "expected #hc.scope to be one of \"WorkGroup\", "
                          "\"SubGroup\", \"WorkItem\"";
  return success();
}

LogicalResult EffectsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  StringRef kind) {
  if (kind != "Pure" && kind != "Read" && kind != "Write" &&
      kind != "ReadWrite")
    return emitError() << "expected #hc.effects to be one of \"Pure\", "
                          "\"Read\", \"Write\", \"ReadWrite\"";
  return success();
}
