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

// Accumulator for the textual fields of a `#hc.layout` attribute.
// `LayoutAttr::parse` drives one field at a time into this; the
// individual flags are checked at the end so a missing field
// produces a single "requires ..." diagnostic instead of a per-
// field one.
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

// Mark `flag` seen, emitting a "duplicate field 'name'" diagnostic at
// `nameLoc` if it had already been set.
static ParseResult markFieldSeen(AsmParser &parser, llvm::SMLoc nameLoc,
                                 StringRef name, bool &flag) {
  if (flag)
    return parser.emitError(nameLoc, "duplicate field '") << name << "'";
  flag = true;
  return success();
}

// Parse one `params = { ... }` value as a DictionaryAttr.
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

// Per-field parsers. Each handles its own value parse but shares the
// duplicate-check + flag-set via `markFieldSeen` on the relevant
// LayoutFields flag.
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

// Table mapping `#hc.layout` field names to their parsers. Sized
// statically so adding a new field is one line; the dispatcher
// walks the array linearly (five entries — branchless on hot paths).
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

// Parse one `<name> = <value>` field inside a `#hc.layout<...>` and
// dispatch onto the matching per-field parser. Unknown names emit at
// `nameLoc`. Returns failure on a malformed field, on a duplicate,
// or on an unknown name.
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

// Validate one `#hc.layout` quoted name list: every entry must be a
// non-empty StringAttr, and each name must be unique across the
// shared `seen` set (shape_syms / index_syms / params keys live in
// the same expression-symbol namespace at substitution time).
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

// Validate the `params` dictionary: non-null, non-empty keys,
// `#hc.expr` values, and key uniqueness against the shared `seen` set.
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

// Product of `dims` for layout-less storage: row-major contiguous, no
// padding, so `storage_size = d_0 * d_1 * ... * d_{n-1}`. Empty dims
// (rank-0) is `1` — that's the convention the layout-driven path also
// observes (`storage_size = 1` for scalar storage), so the two
// branches stay shape-consistent.
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

// Compose `(targets, replacements)` parallel arrays from a quoted
// name list and matching `ExprAttr` replacements. Used to substitute
// `shape_syms` / `index_syms` placeholders for actual operand
// expressions before invoking ixsimpl's `ixs_subs_multi`.
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

// Drive `ixs_subs_multi` on `expr`; returns the substituted handle or
// failure if the engine rejected the input.
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

// Identity row-major offset for a layout-less operand:
// `i_0 * (d_1 * ... * d_{n-1}) + ... + i_{n-1}`. Returns `0` for
// rank-0. Caller verifies rank parity between `dims` and `indexExprs`.
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

// Append one `(target, replacement)` pair, composing the target sym
// handle on the fly. Used by `composeAccessOffsetExpr` to seed both
// the shape-sym and index-sym substitutions into the same pair of
// arrays before one `ixs_subs_multi` call.
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

// Stage the layout-driven substitutions: shape_syms[k] → dims[k],
// index_syms[k] → indexExprs[k]. Caller has already checked rank
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
