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

#include <array>

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

  // `"?"` -> #hc.dyn sentinel: host-owned size, not symbol-derivable.
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

// Flags checked at end -> single "requires ..." diagnostic for any missing
// field.
struct LayoutFields {
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
};

static ParseResult markFieldSeen(AsmParser &parser, llvm::SMLoc nameLoc,
                                 StringRef name, bool &flag) {
  if (flag)
    return parser.emitError(nameLoc, "duplicate field '") << name << "'";
  flag = true;
  return success();
}

static ParseResult parseLayoutParamsField(AsmParser &parser,
                                          DictionaryAttr &out) {
  llvm::SMLoc valueLoc = parser.getCurrentLocation();
  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  out = llvm::dyn_cast<DictionaryAttr>(attr);
  if (!out)
    return parser.emitError(valueLoc,
                            "expected params to be a dictionary attribute");
  return success();
}

static ParseResult parseLayoutShapeSymsField(AsmParser &parser,
                                             LayoutFields &out,
                                             llvm::SMLoc nameLoc) {
  if (failed(markFieldSeen(parser, nameLoc, "shape_syms", out.gotShape)))
    return failure();
  return parseQuotedNameList(parser, out.shapeSyms);
}

static ParseResult parseLayoutIndexSymsField(AsmParser &parser,
                                             LayoutFields &out,
                                             llvm::SMLoc nameLoc) {
  if (failed(markFieldSeen(parser, nameLoc, "index_syms", out.gotIndex)))
    return failure();
  return parseQuotedNameList(parser, out.indexSyms);
}

static ParseResult parseLayoutParamsDispatch(AsmParser &parser,
                                             LayoutFields &out,
                                             llvm::SMLoc nameLoc) {
  if (failed(markFieldSeen(parser, nameLoc, "params", out.gotParams)))
    return failure();
  return parseLayoutParamsField(parser, out.params);
}

static ParseResult parseLayoutStorageSizeField(AsmParser &parser,
                                               LayoutFields &out,
                                               llvm::SMLoc nameLoc) {
  if (failed(markFieldSeen(parser, nameLoc, "storage_size", out.gotStorage)))
    return failure();
  llvm::SMLoc valueLoc = parser.getCurrentLocation();
  return parseTypedAttribute<ExprAttr>(parser, valueLoc, "storage_size",
                                       out.storageSize);
}

static ParseResult parseLayoutOffsetField(AsmParser &parser, LayoutFields &out,
                                          llvm::SMLoc nameLoc) {
  if (failed(markFieldSeen(parser, nameLoc, "offset", out.gotOffset)))
    return failure();
  llvm::SMLoc valueLoc = parser.getCurrentLocation();
  return parseTypedAttribute<ExprAttr>(parser, valueLoc, "offset", out.offset);
}

struct LayoutFieldDispatch {
  StringRef name;
  ParseResult (*parse)(AsmParser &, LayoutFields &, llvm::SMLoc);
};

static const std::array<LayoutFieldDispatch, 5> kLayoutFieldDispatch = {{
    {"shape_syms", &parseLayoutShapeSymsField},
    {"index_syms", &parseLayoutIndexSymsField},
    {"params", &parseLayoutParamsDispatch},
    {"storage_size", &parseLayoutStorageSizeField},
    {"offset", &parseLayoutOffsetField},
}};

static ParseResult parseLayoutField(AsmParser &parser, LayoutFields &out) {
  StringRef name;
  llvm::SMLoc nameLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&name) || parser.parseEqual())
    return failure();
  for (const auto &spec : kLayoutFieldDispatch)
    if (spec.name == name)
      return spec.parse(parser, out, nameLoc);
  return parser.emitError(nameLoc)
         << "unknown #hc.layout field '" << name << "'";
}

} // namespace

Attribute LayoutAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  LayoutFields fields;
  llvm::SMLoc startLoc = parser.getCurrentLocation();

  if (parser.parseCommaSeparatedList(
          [&]() -> ParseResult { return parseLayoutField(parser, fields); }))
    return {};
  if (parser.parseGreater())
    return {};

  if (!fields.gotShape || !fields.gotIndex || !fields.gotParams ||
      !fields.gotStorage || !fields.gotOffset) {
    parser.emitError(startLoc,
                     "#hc.layout requires shape_syms, index_syms, params, "
                     "storage_size, offset");
    return {};
  }

  return LayoutAttr::getChecked([&] { return parser.emitError(startLoc); },
                                parser.getContext(), fields.shapeSyms,
                                fields.indexSyms, fields.params,
                                fields.storageSize, fields.offset);
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

namespace {

// shape_syms / index_syms / params keys share one namespace at substitution.
static LogicalResult
verifyLayoutNameList(function_ref<InFlightDiagnostic()> emitError,
                     ArrayRef<Attribute> names, StringRef field,
                     llvm::DenseSet<StringRef> &seen) {
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
}

static LogicalResult
verifyLayoutParams(function_ref<InFlightDiagnostic()> emitError,
                   DictionaryAttr params, llvm::DenseSet<StringRef> &seen) {
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
  return success();
}

} // namespace

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Attribute> shapeSyms,
                                 ArrayRef<Attribute> indexSyms,
                                 DictionaryAttr params, ExprAttr storageSize,
                                 ExprAttr offset) {
  llvm::DenseSet<StringRef> seen;
  if (failed(verifyLayoutNameList(emitError, shapeSyms, "shape_syms", seen)))
    return failure();
  if (failed(verifyLayoutNameList(emitError, indexSyms, "index_syms", seen)))
    return failure();
  if (shapeSyms.size() != indexSyms.size())
    return emitError() << "shape_syms and index_syms must have the same "
                          "length (got "
                       << shapeSyms.size() << " vs " << indexSyms.size()
                       << "); non-injective storage is expressed through "
                          "`offset` / `storage_size`, not by adding extra "
                          "index_syms";
  if (failed(verifyLayoutParams(emitError, params, seen)))
    return failure();
  if (!storageSize)
    return emitError() << "missing storage_size";
  if (!offset)
    return emitError() << "missing offset";
  return success();
}

namespace {

// Row-major contiguous: prod(dims), empty -> 1 (matches layout-driven scalar
// case).
static FailureOr<sym::ExprHandle>
identityShapeProduct(sym::Store &store, ArrayRef<Attribute> dims) {
  if (dims.empty())
    return sym::composeExprInt(store, 1);
  auto firstExpr = dyn_cast<ExprAttr>(dims.front());
  if (!firstExpr)
    return failure();
  sym::ExprHandle product = firstExpr.getValue();
  for (Attribute dim : dims.drop_front()) {
    auto dimExpr = dyn_cast<ExprAttr>(dim);
    if (!dimExpr)
      return failure();
    auto next = sym::composeExprBinary(store, product, sym::ExprBinaryOp::Mul,
                                       dimExpr.getValue());
    if (failed(next))
      return failure();
    product = *next;
  }
  return product;
}

static LogicalResult
collectSubstitutionPairs(sym::Store &store, ArrayRef<Attribute> nameAttrs,
                         ArrayRef<Attribute> replacementAttrs,
                         SmallVectorImpl<ixs_node *> &targets,
                         SmallVectorImpl<ixs_node *> &replacements) {
  for (auto [sym, replacementAttr] :
       llvm::zip_equal(nameAttrs, replacementAttrs)) {
    auto symStr = dyn_cast<StringAttr>(sym);
    auto replExpr = dyn_cast<ExprAttr>(replacementAttr);
    if (!symStr || !replExpr)
      return failure();
    auto symHandle = sym::composeExprSym(store, symStr.getValue());
    if (failed(symHandle))
      return failure();
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(const_cast<ixs_node *>(replExpr.getValue().raw()));
  }
  return success();
}

static FailureOr<sym::ExprHandle>
substituteMulti(sym::Store &store, sym::ExprHandle expr,
                ArrayRef<ixs_node *> targets,
                ArrayRef<ixs_node *> replacements) {
  sym::Session session(store);
  ixs_node *bound =
      ixs_subs_multi(session.raw(), const_cast<ixs_node *>(expr.raw()),
                     static_cast<uint32_t>(targets.size()),
                     const_cast<ixs_node **>(targets.data()),
                     const_cast<ixs_node **>(replacements.data()));
  if (!bound)
    return failure();
  return sym::ExprHandle(bound);
}

} // namespace

namespace mlir::hc {

mlir::FailureOr<ExprAttr> computeStorageSizeExpr(mlir::MLIRContext *ctx,
                                                 LayoutAttr layout,
                                                 ShapeAttr originalShape) {
  if (!originalShape)
    return failure();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

  if (!layout) {
    auto product = identityShapeProduct(store, originalShape.getDims());
    if (failed(product))
      return failure();
    return ExprAttr::get(ctx, *product);
  }

  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> dims = originalShape.getDims();
  if (shapeSyms.size() != dims.size())
    return failure();

  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(shapeSyms.size());
  replacements.reserve(shapeSyms.size());
  if (failed(collectSubstitutionPairs(store, shapeSyms, dims, targets,
                                      replacements)))
    return failure();

  auto bound = substituteMulti(store, layout.getStorageSize().getValue(),
                               targets, replacements);
  if (failed(bound))
    return failure();
  return ExprAttr::get(ctx, *bound);
}

namespace {

// Row-major: sum_i (i * prod(dims[i+1:])). Rank-0 -> 0. Caller checks rank
// parity.
static FailureOr<sym::ExprHandle>
identityLayoutOffset(sym::Store &store, ArrayRef<ExprAttr> indexExprs,
                     ArrayRef<Attribute> dims) {
  auto zero = sym::composeExprInt(store, 0);
  if (failed(zero))
    return failure();
  if (indexExprs.empty())
    return *zero;
  sym::ExprHandle accum = *zero;
  for (size_t i = 0; i < indexExprs.size(); ++i) {
    sym::ExprHandle term = indexExprs[i].getValue();
    for (size_t j = i + 1; j < dims.size(); ++j) {
      auto dimExpr = llvm::dyn_cast<ExprAttr>(dims[j]);
      if (!dimExpr)
        return failure();
      auto next = sym::composeExprBinary(store, term, sym::ExprBinaryOp::Mul,
                                         dimExpr.getValue());
      if (failed(next))
        return failure();
      term = *next;
    }
    auto added =
        sym::composeExprBinary(store, accum, sym::ExprBinaryOp::Add, term);
    if (failed(added))
      return failure();
    accum = *added;
  }
  return accum;
}

} // namespace

namespace {

static LogicalResult
appendSubstitutionPair(sym::Store &store, StringRef name,
                       sym::ExprHandle replacement,
                       SmallVectorImpl<ixs_node *> &targets,
                       SmallVectorImpl<ixs_node *> &replacements) {
  auto symHandle = sym::composeExprSym(store, name);
  if (failed(symHandle))
    return failure();
  targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
  replacements.push_back(const_cast<ixs_node *>(replacement.raw()));
  return success();
}

// shape_syms[k] -> dims[k]; index_syms[k] -> indexExprs[k]. Caller checks rank
// parity.
static LogicalResult stageLayoutSubstitutions(
    sym::Store &store, ArrayRef<Attribute> shapeSyms, ArrayRef<Attribute> dims,
    ArrayRef<Attribute> indexSyms, ArrayRef<ExprAttr> indexExprs,
    SmallVectorImpl<ixs_node *> &targets,
    SmallVectorImpl<ixs_node *> &replacements) {
  for (auto [sym, dim] : llvm::zip_equal(shapeSyms, dims)) {
    auto dimExpr = llvm::dyn_cast<ExprAttr>(dim);
    if (!dimExpr)
      return failure();
    if (failed(appendSubstitutionPair(
            store, llvm::cast<StringAttr>(sym).getValue(), dimExpr.getValue(),
            targets, replacements)))
      return failure();
  }
  for (auto [sym, idx] : llvm::zip_equal(indexSyms, indexExprs)) {
    if (failed(appendSubstitutionPair(store,
                                      llvm::cast<StringAttr>(sym).getValue(),
                                      idx.getValue(), targets, replacements)))
      return failure();
  }
  return success();
}

} // namespace

mlir::FailureOr<ExprAttr>
composeAccessOffsetExpr(mlir::MLIRContext *ctx, LayoutAttr layout,
                        ShapeAttr originalShape,
                        mlir::ArrayRef<ExprAttr> indexExprs) {
  if (!originalShape)
    return failure();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  ArrayRef<Attribute> dims = originalShape.getDims();

  if (!layout) {
    if (dims.size() != indexExprs.size())
      return failure();
    auto offset = identityLayoutOffset(store, indexExprs, dims);
    if (failed(offset))
      return failure();
    return ExprAttr::get(ctx, *offset);
  }

  ArrayRef<Attribute> shapeSyms = layout.getShapeSyms();
  ArrayRef<Attribute> indexSyms = layout.getIndexSyms();
  if (shapeSyms.size() != dims.size())
    return failure();
  if (indexExprs.size() != indexSyms.size())
    return failure();

  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(shapeSyms.size() + indexSyms.size());
  replacements.reserve(shapeSyms.size() + indexSyms.size());
  if (failed(stageLayoutSubstitutions(store, shapeSyms, dims, indexSyms,
                                      indexExprs, targets, replacements)))
    return failure();

  auto bound = substituteMulti(store, layout.getOffset().getValue(), targets,
                               replacements);
  if (failed(bound))
    return failure();
  return ExprAttr::get(ctx, *bound);
}

} // namespace mlir::hc
