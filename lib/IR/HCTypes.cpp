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

template <typename AttrT>
FailureOr<AttrT> parseTypedAttr(AsmParser &parser, StringRef key,
                                StringRef expected) {
  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  auto typed = dyn_cast<AttrT>(attr);
  if (!typed) {
    parser.emitError(parser.getCurrentLocation())
        << "expected " << expected << " for `" << key << "`";
    return failure();
  }
  return typed;
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

mlir::LogicalResult
mlir::hc::GroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                            ShapeAttr workShape, ShapeAttr groupShape,
                            IntegerAttr subgroupSize) {
  (void)workShape;
  (void)groupShape;
  if (subgroupSize && subgroupSize.getValue().isNegative())
    return emitError() << "subgroup_size must be non-negative";
  return success();
}

mlir::LogicalResult
mlir::hc::WorkitemType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ShapeAttr groupShape, IntegerAttr subgroupSize) {
  (void)groupShape;
  if (subgroupSize && subgroupSize.getValue().isNegative())
    return emitError() << "subgroup_size must be non-negative";
  return success();
}

mlir::LogicalResult
mlir::hc::SubgroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ShapeAttr groupShape, IntegerAttr subgroupSize) {
  (void)groupShape;
  if (subgroupSize && subgroupSize.getValue().isNegative())
    return emitError() << "subgroup_size must be non-negative";
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

Type GroupType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  ShapeAttr workShape;
  ShapeAttr groupShape;
  IntegerAttr subgroupSize;
  if (failed(parser.parseOptionalLess()))
    return GroupType::get(ctx, workShape, groupShape, subgroupSize);

  while (true) {
    StringRef key;
    if (parser.parseKeyword(&key) || parser.parseEqual())
      return {};

    if (key == "work_shape") {
      FailureOr<ShapeAttr> parsed =
          parseTypedAttr<ShapeAttr>(parser, key, "#hc.shape");
      if (failed(parsed))
        return {};
      workShape = *parsed;
    } else if (key == "group_shape") {
      FailureOr<ShapeAttr> parsed =
          parseTypedAttr<ShapeAttr>(parser, key, "#hc.shape");
      if (failed(parsed))
        return {};
      groupShape = *parsed;
    } else if (key == "subgroup_size") {
      FailureOr<IntegerAttr> parsed =
          parseTypedAttr<IntegerAttr>(parser, key, "integer attribute");
      if (failed(parsed))
        return {};
      subgroupSize = *parsed;
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "unknown !hc.group parameter `" << key << "`";
      return {};
    }

    if (failed(parser.parseOptionalComma()))
      break;
  }

  if (parser.parseGreater())
    return {};
  return GroupType::get(ctx, workShape, groupShape, subgroupSize);
}

void GroupType::print(AsmPrinter &printer) const {
  SmallVector<std::pair<StringRef, Attribute>> attrs;
  if (ShapeAttr workShape = getWorkShape())
    attrs.push_back({"work_shape", workShape});
  if (ShapeAttr groupShape = getGroupShape())
    attrs.push_back({"group_shape", groupShape});
  if (IntegerAttr subgroupSize = getSubgroupSize())
    attrs.push_back({"subgroup_size", subgroupSize});
  if (attrs.empty())
    return;

  printer << "<";
  llvm::interleaveComma(attrs, printer, [&](const auto &entry) {
    printer << entry.first << " = ";
    printer.printAttribute(entry.second);
  });
  printer << ">";
}

template <typename TypeT>
static Type parseNestedLaunchContextType(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  ShapeAttr groupShape;
  IntegerAttr subgroupSize;
  if (failed(parser.parseOptionalLess()))
    return TypeT::get(ctx, groupShape, subgroupSize);

  while (true) {
    StringRef key;
    if (parser.parseKeyword(&key) || parser.parseEqual())
      return {};

    if (key == "group_shape") {
      FailureOr<ShapeAttr> parsed =
          parseTypedAttr<ShapeAttr>(parser, key, "#hc.shape");
      if (failed(parsed))
        return {};
      groupShape = *parsed;
    } else if (key == "subgroup_size") {
      FailureOr<IntegerAttr> parsed =
          parseTypedAttr<IntegerAttr>(parser, key, "integer attribute");
      if (failed(parsed))
        return {};
      subgroupSize = *parsed;
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "unknown launch-context parameter `" << key << "`";
      return {};
    }

    if (failed(parser.parseOptionalComma()))
      break;
  }

  if (parser.parseGreater())
    return {};
  return TypeT::get(ctx, groupShape, subgroupSize);
}

static void printNestedLaunchContextType(AsmPrinter &printer,
                                         ShapeAttr groupShape,
                                         IntegerAttr subgroupSize) {
  SmallVector<std::pair<StringRef, Attribute>> attrs;
  if (groupShape)
    attrs.push_back({"group_shape", groupShape});
  if (subgroupSize)
    attrs.push_back({"subgroup_size", subgroupSize});
  if (attrs.empty())
    return;

  printer << "<";
  llvm::interleaveComma(attrs, printer, [&](const auto &entry) {
    printer << entry.first << " = ";
    printer.printAttribute(entry.second);
  });
  printer << ">";
}

Type WorkitemType::parse(AsmParser &parser) {
  return parseNestedLaunchContextType<WorkitemType>(parser);
}

void WorkitemType::print(AsmPrinter &printer) const {
  printNestedLaunchContextType(printer, getGroupShape(), getSubgroupSize());
}

Type SubgroupType::parse(AsmParser &parser) {
  return parseNestedLaunchContextType<SubgroupType>(parser);
}

void SubgroupType::print(AsmPrinter &printer) const {
  printNestedLaunchContextType(printer, getGroupShape(), getSubgroupSize());
}
