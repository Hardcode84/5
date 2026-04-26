// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCTypes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <string>

using namespace mlir;
using namespace mlir::hc;

bool mlir::hc::isHCUndefType(Type type) { return isa<UndefType>(type); }

namespace {

static FailureOr<ShapeAttr> staticShapeFromTupleType(Type shapeType,
                                                     Operation *diagOp) {
  if (!shapeType || isHCUndefType(shapeType)) {
    if (diagOp)
      return diagOp->emitOpError("shape operand is still !hc.undef; "
                                 "expected a concrete tuple of static "
                                 "!hc.idx dimensions");
    return failure();
  }

  auto tuple = dyn_cast<TupleType>(shapeType);
  if (!tuple) {
    if (diagOp)
      return diagOp->emitOpError("shape operand must be a tuple, got ")
             << shapeType;
    return failure();
  }

  SmallVector<Attribute> dims;
  dims.reserve(tuple.size());
  for (auto [idx, dimType] : llvm::enumerate(tuple.getTypes())) {
    auto dim = dyn_cast<IdxType>(dimType);
    if (!dim) {
      if (diagOp)
        return diagOp->emitOpError("shape dimension #")
               << idx << " must be !hc.idx with a static expression, got "
               << dimType;
      return failure();
    }
    if (!dim.getExpr()) {
      if (diagOp)
        return diagOp->emitOpError("shape dimension #")
               << idx << " is dynamic; expected pinned !hc.idx expression";
      return failure();
    }
    dims.push_back(dim.getExpr());
  }
  return ShapeAttr::get(shapeType.getContext(), dims);
}

} // namespace

ShapeAttr mlir::hc::getStaticShapeFromTupleType(Type shapeType) {
  FailureOr<ShapeAttr> shape = staticShapeFromTupleType(shapeType, nullptr);
  return succeeded(shape) ? *shape : ShapeAttr();
}

FailureOr<ShapeAttr>
mlir::hc::verifyStaticShapeFromTupleType(Type shapeType, Operation *diagOp) {
  return staticShapeFromTupleType(shapeType, diagOp);
}

Type mlir::hc::joinHCTypes(Type lhs, Type rhs) {
  if (lhs == rhs)
    return lhs;
  if (isHCUndefType(lhs))
    return rhs;
  if (isHCUndefType(rhs))
    return lhs;
  if (auto joinable = dyn_cast<HCJoinableTypeInterface>(lhs))
    if (Type common = joinable.joinHCType(rhs))
      return common;
  if (auto joinable = dyn_cast<HCJoinableTypeInterface>(rhs))
    if (Type common = joinable.joinHCType(lhs))
      return common;

  auto lhsTuple = dyn_cast<TupleType>(lhs);
  auto rhsTuple = dyn_cast<TupleType>(rhs);
  if (!lhsTuple || !rhsTuple || lhsTuple.size() != rhsTuple.size())
    return {};
  SmallVector<Type> elements;
  elements.reserve(lhsTuple.size());
  for (auto [lhsElement, rhsElement] :
       llvm::zip_equal(lhsTuple.getTypes(), rhsTuple.getTypes())) {
    Type joined = joinHCTypes(lhsElement, rhsElement);
    if (!joined)
      return {};
    elements.push_back(joined);
  }
  return TupleType::get(lhs.getContext(), elements);
}

bool mlir::hc::areHCProgressiveTypesCompatible(Type source, Type dest) {
  if (isHCUndefType(source) || isHCUndefType(dest) || source == dest)
    return true;
  auto sourceTuple = dyn_cast<TupleType>(source);
  auto destTuple = dyn_cast<TupleType>(dest);
  if (!sourceTuple || !destTuple || sourceTuple.size() != destTuple.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(sourceTuple.getTypes(), destTuple.getTypes()),
      [](auto pair) {
        auto [sourceElem, destElem] = pair;
        return areHCProgressiveTypesCompatible(sourceElem, destElem);
      });
}

bool mlir::hc::areHCBranchTypesCompatible(Type source, Type dest) {
  return static_cast<bool>(joinHCTypes(source, dest));
}

bool mlir::hc::shouldRefineHCType(Type current, Type inferred) {
  if (!inferred || current == inferred)
    return false;
  if (isHCUndefType(current))
    return true;
  if (auto currentTuple = dyn_cast<TupleType>(current)) {
    auto inferredTuple = dyn_cast<TupleType>(inferred);
    if (!inferredTuple || currentTuple.size() != inferredTuple.size())
      return false;
    return llvm::any_of(
        llvm::zip_equal(currentTuple.getTypes(), inferredTuple.getTypes()),
        [](auto pair) {
          auto [currentElement, inferredElement] = pair;
          return shouldRefineHCType(currentElement, inferredElement);
        });
  }
  if (auto currentIdx = dyn_cast<IdxType>(current)) {
    auto inferredIdx = dyn_cast<IdxType>(inferred);
    return inferredIdx && !currentIdx.getExpr() && inferredIdx.getExpr();
  }
  if (auto currentPred = dyn_cast<PredType>(current)) {
    auto inferredPred = dyn_cast<PredType>(inferred);
    return inferredPred && !currentPred.getPred() && inferredPred.getPred();
  }
  return false;
}

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

LaunchGeoMethodInfo mlir::hc::getLaunchGeoMethodInfo(LaunchGeoMethod method) {
  switch (method) {
  case LaunchGeoMethod::GroupId:
    return LaunchGeoMethodInfo{LaunchGeoMethod::GroupId, "group_id", "$WG",
                               LaunchGeoArity::MultiAxis,
                               LaunchGeoRankDomain::WorkGridWithGroupFallback};
  case LaunchGeoMethod::LocalId:
    return LaunchGeoMethodInfo{LaunchGeoMethod::LocalId, "local_id", "$WI",
                               LaunchGeoArity::MultiAxis,
                               LaunchGeoRankDomain::Workgroup};
  case LaunchGeoMethod::SubgroupId:
    return LaunchGeoMethodInfo{LaunchGeoMethod::SubgroupId, "subgroup_id",
                               "$SG", LaunchGeoArity::MultiAxis,
                               LaunchGeoRankDomain::Workgroup};
  case LaunchGeoMethod::GroupShape:
    return LaunchGeoMethodInfo{LaunchGeoMethod::GroupShape, "group_shape",
                               "$WGS", LaunchGeoArity::MultiAxis,
                               LaunchGeoRankDomain::Workgroup};
  case LaunchGeoMethod::WorkOffset:
    return LaunchGeoMethodInfo{LaunchGeoMethod::WorkOffset, "work_offset",
                               "$WO", LaunchGeoArity::MultiAxis,
                               LaunchGeoRankDomain::WorkGrid};
  case LaunchGeoMethod::WorkShape:
    return LaunchGeoMethodInfo{LaunchGeoMethod::WorkShape, "work_shape", "$WS",
                               LaunchGeoArity::MultiAxis,
                               LaunchGeoRankDomain::WorkGrid};
  case LaunchGeoMethod::GroupSize:
    return LaunchGeoMethodInfo{LaunchGeoMethod::GroupSize, "group_size", "$GSZ",
                               LaunchGeoArity::Scalar,
                               LaunchGeoRankDomain::Scalar};
  case LaunchGeoMethod::WaveSize:
    return LaunchGeoMethodInfo{LaunchGeoMethod::WaveSize, "wave_size", "$WV",
                               LaunchGeoArity::Scalar,
                               LaunchGeoRankDomain::Scalar};
  }
  llvm_unreachable("unhandled launch-geometry method");
}

std::optional<LaunchGeoMethodInfo>
mlir::hc::classifyLaunchGeoMethod(StringRef method) {
  enum class ClassifiedMethod {
    Unknown,
    GroupId,
    LocalId,
    SubgroupId,
    GroupShape,
    WorkOffset,
    WorkShape,
    GroupSize,
    WaveSize,
  };
  ClassifiedMethod kind = llvm::StringSwitch<ClassifiedMethod>(method)
                              .Case("group_id", ClassifiedMethod::GroupId)
                              .Case("local_id", ClassifiedMethod::LocalId)
                              .Case("subgroup_id", ClassifiedMethod::SubgroupId)
                              .Case("group_shape", ClassifiedMethod::GroupShape)
                              .Case("work_offset", ClassifiedMethod::WorkOffset)
                              .Case("work_shape", ClassifiedMethod::WorkShape)
                              .Case("group_size", ClassifiedMethod::GroupSize)
                              .Case("wave_size", ClassifiedMethod::WaveSize)
                              .Default(ClassifiedMethod::Unknown);
  switch (kind) {
  case ClassifiedMethod::GroupId:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::GroupId);
  case ClassifiedMethod::LocalId:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::LocalId);
  case ClassifiedMethod::SubgroupId:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::SubgroupId);
  case ClassifiedMethod::GroupShape:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::GroupShape);
  case ClassifiedMethod::WorkOffset:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::WorkOffset);
  case ClassifiedMethod::WorkShape:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::WorkShape);
  case ClassifiedMethod::GroupSize:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::GroupSize);
  case ClassifiedMethod::WaveSize:
    return getLaunchGeoMethodInfo(LaunchGeoMethod::WaveSize);
  case ClassifiedMethod::Unknown:
    return std::nullopt;
  }
  llvm_unreachable("unhandled launch-geometry method");
}

std::optional<LaunchContextMetadata>
mlir::hc::getLaunchContextMetadata(Type contextType) {
  if (auto group = dyn_cast_or_null<GroupType>(contextType))
    return LaunchContextMetadata{group.getWorkShape(), group.getGroupShape(),
                                 group.getSubgroupSize()};
  // Nested launch contexts intentionally carry only workgroup-local metadata;
  // work-grid queries must come from the enclosing group handle.
  if (auto workitem = dyn_cast_or_null<WorkitemType>(contextType))
    return LaunchContextMetadata{/*workShape=*/ShapeAttr(),
                                 workitem.getGroupShape(),
                                 workitem.getSubgroupSize()};
  if (auto subgroup = dyn_cast_or_null<SubgroupType>(contextType))
    return LaunchContextMetadata{/*workShape=*/ShapeAttr(),
                                 subgroup.getGroupShape(),
                                 subgroup.getSubgroupSize()};
  return std::nullopt;
}

mlir::LogicalResult
mlir::hc::BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

ShapeAttr mlir::hc::BufferType::getSymbolicShape() const { return getShape(); }
Type mlir::hc::BufferType::getSymbolicElementType() const {
  return getElementType();
}

mlir::LogicalResult
mlir::hc::TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

ShapeAttr mlir::hc::TensorType::getSymbolicShape() const { return getShape(); }
Type mlir::hc::TensorType::getSymbolicElementType() const {
  return getElementType();
}

mlir::LogicalResult
mlir::hc::VectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape) {
  (void)elementType;
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

ShapeAttr mlir::hc::VectorType::getSymbolicShape() const { return getShape(); }
Type mlir::hc::VectorType::getSymbolicElementType() const {
  return getElementType();
}

mlir::LogicalResult
mlir::hc::GroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                            ShapeAttr workShape, ShapeAttr groupShape,
                            ExprAttr subgroupSize) {
  (void)workShape;
  (void)groupShape;
  if (subgroupSize) {
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(subgroupSize.getValue());
    if (value && *value < 0)
      return emitError() << "subgroup_size must be non-negative";
  }
  return success();
}

mlir::LogicalResult
mlir::hc::WorkitemType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ShapeAttr groupShape, ExprAttr subgroupSize) {
  (void)groupShape;
  if (subgroupSize) {
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(subgroupSize.getValue());
    if (value && *value < 0)
      return emitError() << "subgroup_size must be non-negative";
  }
  return success();
}

mlir::LogicalResult
mlir::hc::SubgroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ShapeAttr groupShape, ExprAttr subgroupSize) {
  (void)groupShape;
  if (subgroupSize) {
    std::optional<int64_t> value =
        sym::getIntegerLiteralValue(subgroupSize.getValue());
    if (value && *value < 0)
      return emitError() << "subgroup_size must be non-negative";
  }
  return success();
}

static FailureOr<ExprAttr> parseSubgroupSizeAttr(AsmParser &parser,
                                                 StringRef key) {
  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  if (auto expr = dyn_cast<ExprAttr>(attr))
    return expr;
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    SmallString<32> text;
    integer.getValue().toStringSigned(text);
    auto *dialect = parser.getContext()->getOrLoadDialect<HCDialect>();
    std::string diagnostic;
    FailureOr<sym::ExprHandle> handle =
        sym::parseExpr(dialect->getSymbolStore(), text, &diagnostic);
    if (failed(handle)) {
      parser.emitError(parser.getCurrentLocation(),
                       diagnostic.empty() ? "invalid subgroup_size expression"
                                          : diagnostic);
      return failure();
    }
    return ExprAttr::get(parser.getContext(), *handle);
  }
  parser.emitError(parser.getCurrentLocation())
      << "expected #hc.expr or integer attribute for `" << key << "`";
  return failure();
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

static FailureOr<Type> parseSlicePartType(AsmParser &parser, StringRef key) {
  Type type;
  if (parser.parseType(type))
    return failure();
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "expected type for `" << key << "`";
    return failure();
  }
  return type;
}

Type SliceType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  Type lowerType;
  Type upperType;
  Type stepType;
  if (failed(parser.parseOptionalLess()))
    return SliceType::get(ctx, lowerType, upperType, stepType);

  while (true) {
    StringRef key;
    if (parser.parseKeyword(&key) || parser.parseEqual())
      return {};

    if (key == "lower") {
      FailureOr<Type> parsed = parseSlicePartType(parser, key);
      if (failed(parsed))
        return {};
      lowerType = *parsed;
    } else if (key == "upper") {
      FailureOr<Type> parsed = parseSlicePartType(parser, key);
      if (failed(parsed))
        return {};
      upperType = *parsed;
    } else if (key == "step") {
      FailureOr<Type> parsed = parseSlicePartType(parser, key);
      if (failed(parsed))
        return {};
      stepType = *parsed;
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "unknown !hc.slice parameter `" << key << "`";
      return {};
    }

    if (failed(parser.parseOptionalComma()))
      break;
  }

  if (parser.parseGreater())
    return {};
  return SliceType::get(ctx, lowerType, upperType, stepType);
}

void SliceType::print(AsmPrinter &printer) const {
  SmallVector<std::pair<StringRef, Type>> parts;
  if (Type lower = getLowerType())
    parts.push_back({"lower", lower});
  if (Type upper = getUpperType())
    parts.push_back({"upper", upper});
  if (Type step = getStepType())
    parts.push_back({"step", step});
  if (parts.empty())
    return;

  printer << "<";
  llvm::interleaveComma(parts, printer, [&](const auto &entry) {
    printer << entry.first << " = " << entry.second;
  });
  printer << ">";
}

static Type joinSlicePart(Type lhs, Type rhs, bool &ok) {
  if (!lhs && !rhs)
    return {};
  if (!lhs || !rhs) {
    ok = false;
    return {};
  }
  Type joined = joinHCTypes(lhs, rhs);
  if (!joined)
    ok = false;
  return joined;
}

Type SliceType::joinHCType(Type other) const {
  auto rhs = dyn_cast<SliceType>(other);
  if (!rhs)
    return {};

  bool ok = true;
  Type lower = joinSlicePart(getLowerType(), rhs.getLowerType(), ok);
  Type upper = joinSlicePart(getUpperType(), rhs.getUpperType(), ok);
  Type step = joinSlicePart(getStepType(), rhs.getStepType(), ok);
  if (!ok)
    return SliceType::get(getContext(), Type{}, Type{}, Type{});
  return SliceType::get(getContext(), lower, upper, step);
}

Type GroupType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  SMLoc typeLoc = parser.getCurrentLocation();
  ShapeAttr workShape;
  ShapeAttr groupShape;
  ExprAttr subgroupSize;
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
      FailureOr<ExprAttr> parsed = parseSubgroupSizeAttr(parser, key);
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
  return GroupType::getChecked([&] { return parser.emitError(typeLoc); }, ctx,
                               workShape, groupShape, subgroupSize);
}

void GroupType::print(AsmPrinter &printer) const {
  SmallVector<std::pair<StringRef, Attribute>> attrs;
  if (ShapeAttr workShape = getWorkShape())
    attrs.push_back({"work_shape", workShape});
  if (ShapeAttr groupShape = getGroupShape())
    attrs.push_back({"group_shape", groupShape});
  if (ExprAttr subgroupSize = getSubgroupSize())
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
  SMLoc typeLoc = parser.getCurrentLocation();
  ShapeAttr groupShape;
  ExprAttr subgroupSize;
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
      FailureOr<ExprAttr> parsed = parseSubgroupSizeAttr(parser, key);
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
  return TypeT::getChecked([&] { return parser.emitError(typeLoc); }, ctx,
                           groupShape, subgroupSize);
}

static void printNestedLaunchContextType(AsmPrinter &printer,
                                         ShapeAttr groupShape,
                                         ExprAttr subgroupSize) {
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
