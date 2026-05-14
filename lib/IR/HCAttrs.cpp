// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCAttrs.h"

#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::hc;

#include "hc/IR/HCEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/IR/HCAttrs.cpp.inc"

namespace {

static ParseResult parseShapeDim(AsmParser &parser,
                                 SmallVectorImpl<Attribute> &dims) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  std::string text;
  OptionalParseResult parsedString = parser.parseOptionalString(&text);
  if (!parsedString.has_value())
    return parser.emitError(loc) << "expected quoted hc.shape dimension";
  if (failed(*parsedString))
    return failure();

  // The lone `"?"` spelling lifts a `#hc.dyn` sentinel into the shape
  // entry — used for buffer 1D-collapse and any other shape slot whose
  // size is host-owned / not derivable from the in-IR symbol set.
  if (text == "?") {
    dims.push_back(DynSizeAttr::get(parser.getContext()));
    return success();
  }

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

static FailureOr<ShapeAttr> parseShapeDims(AsmParser &parser,
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

static void printShapeDims(AsmPrinter &printer, ShapeAttr shape) {
  auto &store =
      shape.getContext()->getOrLoadDialect<HCDialect>()->getSymbolStore();
  printer << "[";
  llvm::interleaveComma(shape.getDims(), printer, [&](Attribute dim) {
    if (llvm::isa<DynSizeAttr>(dim)) {
      printer.printString("?");
      return;
    }
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
    if (!llvm::isa<ExprAttr, DynSizeAttr>(dim))
      return emitError()
             << "expected shape dims to be #hc.expr or #hc.dyn attributes";
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

namespace {

static ParseResult parseQuotedNameList(AsmParser &parser,
                                       SmallVectorImpl<Attribute> &out) {
  if (parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  auto parseOne = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();
    std::string text;
    OptionalParseResult parsed = parser.parseOptionalString(&text);
    if (!parsed.has_value())
      return parser.emitError(loc, "expected quoted name");
    if (failed(*parsed))
      return failure();
    out.push_back(StringAttr::get(parser.getContext(), text));
    return success();
  };
  if (parseOne())
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    if (parseOne())
      return failure();
  }
  return parser.parseRSquare();
}

static void printQuotedNameList(AsmPrinter &printer,
                                ArrayRef<Attribute> names) {
  printer << "[";
  llvm::interleaveComma(names, printer, [&](Attribute a) {
    printer.printString(llvm::cast<StringAttr>(a).getValue());
  });
  printer << "]";
}

template <typename AttrT>
static ParseResult parseTypedAttribute(AsmParser &parser, llvm::SMLoc loc,
                                       StringRef field, AttrT &out) {
  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  out = llvm::dyn_cast<AttrT>(attr);
  if (!out)
    return parser.emitError(loc) << "expected " << field << " to be a "
                                 << AttrT::getMnemonic() << " attribute";
  return success();
}

} // namespace

Attribute LayoutAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  SmallVector<Attribute> shapeSyms;
  SmallVector<Attribute> indexSyms;
  DictionaryAttr params;
  ExprAttr storageSize;
  ExprAttr offset;
  bool gotShape = false;
  bool gotIndex = false;
  bool gotParams = false;
  bool gotStorage = false;
  bool gotOffset = false;
  llvm::SMLoc startLoc = parser.getCurrentLocation();

  auto parseField = [&]() -> ParseResult {
    StringRef name;
    llvm::SMLoc nameLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&name) || parser.parseEqual())
      return failure();
    if (name == "shape_syms") {
      if (gotShape)
        return parser.emitError(nameLoc, "duplicate field 'shape_syms'");
      gotShape = true;
      return parseQuotedNameList(parser, shapeSyms);
    }
    if (name == "index_syms") {
      if (gotIndex)
        return parser.emitError(nameLoc, "duplicate field 'index_syms'");
      gotIndex = true;
      return parseQuotedNameList(parser, indexSyms);
    }
    if (name == "params") {
      if (gotParams)
        return parser.emitError(nameLoc, "duplicate field 'params'");
      gotParams = true;
      llvm::SMLoc valueLoc = parser.getCurrentLocation();
      Attribute attr;
      if (parser.parseAttribute(attr))
        return failure();
      params = llvm::dyn_cast<DictionaryAttr>(attr);
      if (!params)
        return parser.emitError(valueLoc,
                                "expected params to be a dictionary attribute");
      return success();
    }
    if (name == "storage_size") {
      if (gotStorage)
        return parser.emitError(nameLoc, "duplicate field 'storage_size'");
      gotStorage = true;
      llvm::SMLoc valueLoc = parser.getCurrentLocation();
      return parseTypedAttribute<ExprAttr>(parser, valueLoc, "storage_size",
                                           storageSize);
    }
    if (name == "offset") {
      if (gotOffset)
        return parser.emitError(nameLoc, "duplicate field 'offset'");
      gotOffset = true;
      llvm::SMLoc valueLoc = parser.getCurrentLocation();
      return parseTypedAttribute<ExprAttr>(parser, valueLoc, "offset", offset);
    }
    return parser.emitError(nameLoc)
           << "unknown #hc.layout field '" << name << "'";
  };

  if (parser.parseCommaSeparatedList(parseField))
    return {};
  if (parser.parseGreater())
    return {};

  if (!gotShape || !gotIndex || !gotParams || !gotStorage || !gotOffset) {
    parser.emitError(startLoc,
                     "#hc.layout requires shape_syms, index_syms, params, "
                     "storage_size, offset");
    return {};
  }

  return LayoutAttr::getChecked([&] { return parser.emitError(startLoc); },
                                parser.getContext(), shapeSyms, indexSyms,
                                params, storageSize, offset);
}

void LayoutAttr::print(AsmPrinter &printer) const {
  printer << "<shape_syms = ";
  printQuotedNameList(printer, getShapeSyms());
  printer << ", index_syms = ";
  printQuotedNameList(printer, getIndexSyms());
  printer << ", params = " << getParams();
  printer << ", storage_size = " << getStorageSize();
  printer << ", offset = " << getOffset();
  printer << ">";
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Attribute> shapeSyms,
                                 ArrayRef<Attribute> indexSyms,
                                 DictionaryAttr params, ExprAttr storageSize,
                                 ExprAttr offset) {
  // Single shared set: shape_syms / index_syms / params keys live in the
  // same expression-symbol namespace at substitution time, so their names
  // must be pairwise disjoint.
  llvm::DenseSet<StringRef> seen;
  auto checkNameList = [&](ArrayRef<Attribute> names,
                           StringRef field) -> LogicalResult {
    for (Attribute attr : names) {
      auto str = llvm::dyn_cast_if_present<StringAttr>(attr);
      if (!str)
        return emitError() << "expected " << field
                           << " entries to be string attributes";
      if (str.getValue().empty())
        return emitError() << field << " entries must be non-empty";
      if (!seen.insert(str.getValue()).second)
        return emitError() << "duplicate name '" << str.getValue()
                           << "' across #hc.layout name lists";
    }
    return success();
  };
  if (failed(checkNameList(shapeSyms, "shape_syms")))
    return failure();
  if (failed(checkNameList(indexSyms, "index_syms")))
    return failure();
  if (shapeSyms.size() != indexSyms.size())
    return emitError() << "shape_syms and index_syms must have the same "
                          "length (got "
                       << shapeSyms.size() << " vs " << indexSyms.size()
                       << "); non-injective storage is expressed through "
                          "`offset` / `storage_size`, not by adding extra "
                          "index_syms";
  if (!params)
    return emitError() << "expected params to be a non-null dictionary";
  for (NamedAttribute kv : params) {
    StringRef key = kv.getName().getValue();
    if (key.empty())
      return emitError() << "params keys must be non-empty";
    if (!llvm::isa<ExprAttr>(kv.getValue()))
      return emitError() << "params value for '" << key
                         << "' must be a #hc.expr attribute";
    if (!seen.insert(key).second)
      return emitError() << "duplicate name '" << key
                         << "' across #hc.layout name lists";
  }
  if (!storageSize)
    return emitError() << "missing storage_size";
  if (!offset)
    return emitError() << "missing offset";
  return success();
}
