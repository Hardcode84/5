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

// Pinned `!hc.idx<expr>` only. diagOp non-null → emit error; null → silent.
static LogicalResult collectTupleDim(Type dimType, size_t idx,
                                     Operation *diagOp,
                                     SmallVectorImpl<Attribute> &dims) {
  auto dim = dyn_cast<IdxType>(dimType);
  if (!dim) {
    if (diagOp)
      diagOp->emitOpError("shape dimension #")
          << idx << " must be !hc.idx with a static expression, got "
          << dimType;
    return failure();
  }
  if (!dim.getExpr()) {
    if (diagOp)
      diagOp->emitOpError("shape dimension #")
          << idx << " is dynamic; expected pinned !hc.idx expression";
    return failure();
  }
  dims.push_back(dim.getExpr());
  return success();
}

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
  for (auto [idx, dimType] : llvm::enumerate(tuple.getTypes()))
    if (failed(collectTupleDim(dimType, idx, diagOp, dims)))
      return failure();
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

// Result in `lhs`'s context.
static Type joinHCTupleTypes(TupleType lhs, TupleType rhs) {
  if (lhs.size() != rhs.size())
    return {};
  SmallVector<Type> elements;
  elements.reserve(lhs.size());
  for (auto [lhsElement, rhsElement] :
       llvm::zip_equal(lhs.getTypes(), rhs.getTypes())) {
    Type joined = mlir::hc::joinHCTypes(lhsElement, rhsElement);
    if (!joined)
      return {};
    elements.push_back(joined);
  }
  return TupleType::get(lhs.getContext(), elements);
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
  if (!lhsTuple || !rhsTuple)
    return {};
  return joinHCTupleTypes(lhsTuple, rhsTuple);
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

static bool shouldRefineHCTuple(TupleType current, TupleType inferred) {
  if (current.size() != inferred.size())
    return false;
  return llvm::any_of(
      llvm::zip_equal(current.getTypes(), inferred.getTypes()), [](auto pair) {
        auto [currentElement, inferredElement] = pair;
        return mlir::hc::shouldRefineHCType(currentElement, inferredElement);
      });
}

// Pinned inferred refines unpinned current of the same kind.
static bool shouldRefineHCIdx(IdxType current, Type inferred) {
  auto inferredIdx = dyn_cast<IdxType>(inferred);
  return inferredIdx && !current.getExpr() && inferredIdx.getExpr();
}

static bool shouldRefineHCPred(PredType current, Type inferred) {
  auto inferredPred = dyn_cast<PredType>(inferred);
  return inferredPred && !current.getPred() && inferredPred.getPred();
}

bool mlir::hc::shouldRefineHCType(Type current, Type inferred) {
  if (!inferred || current == inferred)
    return false;
  if (isHCUndefType(current))
    return true;
  if (auto currentTuple = dyn_cast<TupleType>(current)) {
    auto inferredTuple = dyn_cast<TupleType>(inferred);
    return inferredTuple && shouldRefineHCTuple(currentTuple, inferredTuple);
  }
  if (auto currentIdx = dyn_cast<IdxType>(current))
    return shouldRefineHCIdx(currentIdx, inferred);
  if (auto currentPred = dyn_cast<PredType>(current))
    return shouldRefineHCPred(currentPred, inferred);
  return false;
}

namespace {

// `!hc.idx` / `!hc.pred` inline as quoted string via dialect-owned ixsimpl
// store.

template <typename HandleT>
using StoreParser = FailureOr<HandleT> (*)(sym::Store &, llvm::StringRef,
                                           std::string *);

// Accepts inline `"expr"` or attribute `#hc.expr<"expr">`.
template <typename AttrT, typename HandleT, StoreParser<HandleT> ParseFn>
static FailureOr<AttrT> parseInlineOrAttrForm(AsmParser &parser,
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

static void printInlineNode(AsmPrinter &printer, MLIRContext *ctx,
                            const ixs_node *node) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  printer.printString(store.render(node));
}

template <typename AttrT>
static FailureOr<AttrT> parseTypedAttr(AsmParser &parser, StringRef key,
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
  // Nested contexts: workgroup-local only; work-grid lives on enclosing group.
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

// Shape non-null. Layout self-checks. Rank vs shape_syms parity in later
// passes.
static mlir::LogicalResult
verifyShapedTypeShell(function_ref<InFlightDiagnostic()> emitError,
                      ShapeAttr shape, LayoutAttr /*layout*/) {
  if (!shape)
    return emitError() << "expected #hc.shape attribute";
  return success();
}

mlir::LogicalResult
mlir::hc::BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape,
                             LayoutAttr layout) {
  (void)elementType;
  return verifyShapedTypeShell(emitError, shape, layout);
}

ShapeAttr mlir::hc::BufferType::getSymbolicShape() const { return getShape(); }
Type mlir::hc::BufferType::getSymbolicElementType() const {
  return getElementType();
}
LayoutAttr mlir::hc::BufferType::getSymbolicLayout() const {
  return getLayout();
}
Type mlir::hc::BufferType::cloneWithSymbolicLayout(LayoutAttr layout) const {
  return BufferType::get(getContext(), getElementType(), getShape(), layout);
}
Type mlir::hc::BufferType::cloneWithSymbolicShape(ShapeAttr shape) const {
  return BufferType::get(getContext(), getElementType(), shape, getLayout());
}

namespace {

// Same flavor/element/shape. Bare ∪ layout → layout. Two distinct layouts →
// fail.
template <typename ShapedT>
static Type joinShapedSameFlavor(ShapedT lhs, Type rhsRaw) {
  auto rhs = dyn_cast<ShapedT>(rhsRaw);
  if (!rhs)
    return {};
  if (lhs.getElementType() != rhs.getElementType())
    return {};
  if (lhs.getShape() != rhs.getShape())
    return {};
  LayoutAttr lhsLayout = lhs.getLayout();
  LayoutAttr rhsLayout = rhs.getLayout();
  if (lhsLayout == rhsLayout)
    return lhs;
  if (!lhsLayout)
    return rhs;
  if (!rhsLayout)
    return lhs;
  return {};
}

} // namespace

Type mlir::hc::BufferType::joinHCType(Type other) const {
  return joinShapedSameFlavor(*this, other);
}

mlir::LogicalResult
mlir::hc::TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape,
                             LayoutAttr layout) {
  (void)elementType;
  return verifyShapedTypeShell(emitError, shape, layout);
}

ShapeAttr mlir::hc::TensorType::getSymbolicShape() const { return getShape(); }
Type mlir::hc::TensorType::getSymbolicElementType() const {
  return getElementType();
}
LayoutAttr mlir::hc::TensorType::getSymbolicLayout() const {
  return getLayout();
}
Type mlir::hc::TensorType::cloneWithSymbolicLayout(LayoutAttr layout) const {
  return TensorType::get(getContext(), getElementType(), getShape(), layout);
}
Type mlir::hc::TensorType::cloneWithSymbolicShape(ShapeAttr shape) const {
  return TensorType::get(getContext(), getElementType(), shape, getLayout());
}

Type mlir::hc::TensorType::joinHCType(Type other) const {
  return joinShapedSameFlavor(*this, other);
}

mlir::LogicalResult
mlir::hc::VectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType, ShapeAttr shape,
                             LayoutAttr layout) {
  (void)elementType;
  return verifyShapedTypeShell(emitError, shape, layout);
}

ShapeAttr mlir::hc::VectorType::getSymbolicShape() const { return getShape(); }
Type mlir::hc::VectorType::getSymbolicElementType() const {
  return getElementType();
}
LayoutAttr mlir::hc::VectorType::getSymbolicLayout() const {
  return getLayout();
}
Type mlir::hc::VectorType::cloneWithSymbolicLayout(LayoutAttr layout) const {
  return VectorType::get(getContext(), getElementType(), getShape(), layout);
}
Type mlir::hc::VectorType::cloneWithSymbolicShape(ShapeAttr shape) const {
  return VectorType::get(getContext(), getElementType(), shape, getLayout());
}

Type mlir::hc::VectorType::joinHCType(Type other) const {
  return joinShapedSameFlavor(*this, other);
}

mlir::LogicalResult
mlir::hc::BareTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type elementType, ShapeAttr shape,
                                 LayoutAttr layout) {
  (void)elementType;
  return verifyShapedTypeShell(emitError, shape, layout);
}

ShapeAttr mlir::hc::BareTensorType::getSymbolicShape() const {
  return getShape();
}
Type mlir::hc::BareTensorType::getSymbolicElementType() const {
  return getElementType();
}
LayoutAttr mlir::hc::BareTensorType::getSymbolicLayout() const {
  return getLayout();
}
Type mlir::hc::BareTensorType::cloneWithSymbolicLayout(
    LayoutAttr layout) const {
  return BareTensorType::get(getContext(), getElementType(), getShape(),
                             layout);
}
Type mlir::hc::BareTensorType::cloneWithSymbolicShape(ShapeAttr shape) const {
  return BareTensorType::get(getContext(), getElementType(), shape,
                             getLayout());
}

Type mlir::hc::BareTensorType::joinHCType(Type other) const {
  return joinShapedSameFlavor(*this, other);
}

mlir::LogicalResult
mlir::hc::BareVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type elementType, ShapeAttr shape,
                                 LayoutAttr layout) {
  (void)elementType;
  return verifyShapedTypeShell(emitError, shape, layout);
}

ShapeAttr mlir::hc::BareVectorType::getSymbolicShape() const {
  return getShape();
}
Type mlir::hc::BareVectorType::getSymbolicElementType() const {
  return getElementType();
}
LayoutAttr mlir::hc::BareVectorType::getSymbolicLayout() const {
  return getLayout();
}
Type mlir::hc::BareVectorType::cloneWithSymbolicLayout(
    LayoutAttr layout) const {
  return BareVectorType::get(getContext(), getElementType(), getShape(),
                             layout);
}
Type mlir::hc::BareVectorType::cloneWithSymbolicShape(ShapeAttr shape) const {
  return BareVectorType::get(getContext(), getElementType(), shape,
                             getLayout());
}

Type mlir::hc::BareVectorType::joinHCType(Type other) const {
  return joinShapedSameFlavor(*this, other);
}

// Absent = unknown. Zero pins IdxType<#hc.expr<"0">> on hc.wave_size via
// firstDim / 0.
static mlir::LogicalResult
verifySubgroupSize(function_ref<InFlightDiagnostic()> emitError,
                   ExprAttr subgroupSize) {
  if (!subgroupSize)
    return success();
  std::optional<int64_t> value =
      sym::getIntegerLiteralValue(subgroupSize.getValue());
  if (value && *value <= 0)
    return emitError() << "subgroup_size must be positive";
  return success();
}

mlir::LogicalResult
mlir::hc::GroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                            ShapeAttr workShape, ShapeAttr groupShape,
                            ExprAttr subgroupSize) {
  (void)workShape;
  (void)groupShape;
  return verifySubgroupSize(emitError, subgroupSize);
}

mlir::LogicalResult
mlir::hc::WorkitemType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ShapeAttr groupShape, ExprAttr subgroupSize) {
  (void)groupShape;
  return verifySubgroupSize(emitError, subgroupSize);
}

mlir::LogicalResult
mlir::hc::SubgroupType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ShapeAttr groupShape, ExprAttr subgroupSize) {
  (void)groupShape;
  return verifySubgroupSize(emitError, subgroupSize);
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
  // Distinct symbolic facts widen to unpinned — concretizing guesses a
  // predecessor.
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
  // Widen: keep kind, drop payload.
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

// Caller already saw `<`. Consumes trailing `>`. Unknown keys diagnosed by
// handleField.
template <typename FieldHandler>
static LogicalResult parseAngleBracketedFields(AsmParser &parser,
                                               FieldHandler handleField) {
  while (true) {
    StringRef key;
    if (parser.parseKeyword(&key) || parser.parseEqual())
      return failure();
    if (failed(handleField(key)))
      return failure();
    if (failed(parser.parseOptionalComma()))
      break;
  }
  return parser.parseGreater();
}

static LogicalResult parseSliceField(AsmParser &parser, StringRef key,
                                     Type &lowerType, Type &upperType,
                                     Type &stepType) {
  Type *target = nullptr;
  if (key == "lower")
    target = &lowerType;
  else if (key == "upper")
    target = &upperType;
  else if (key == "step")
    target = &stepType;
  if (!target) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown !hc.slice parameter `" << key << "`";
    return failure();
  }
  FailureOr<Type> parsed = parseSlicePartType(parser, key);
  if (failed(parsed))
    return failure();
  *target = *parsed;
  return success();
}

Type SliceType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  Type lowerType, upperType, stepType;
  if (failed(parser.parseOptionalLess()))
    return SliceType::get(ctx, lowerType, upperType, stepType);
  if (failed(parseAngleBracketedFields(parser, [&](StringRef key) {
        return parseSliceField(parser, key, lowerType, upperType, stepType);
      })))
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

static LogicalResult parseGroupField(AsmParser &parser, StringRef key,
                                     ShapeAttr &workShape,
                                     ShapeAttr &groupShape,
                                     ExprAttr &subgroupSize) {
  if (key == "work_shape" || key == "group_shape") {
    FailureOr<ShapeAttr> parsed =
        parseTypedAttr<ShapeAttr>(parser, key, "#hc.shape");
    if (failed(parsed))
      return failure();
    (key == "work_shape" ? workShape : groupShape) = *parsed;
    return success();
  }
  if (key == "subgroup_size") {
    FailureOr<ExprAttr> parsed = parseSubgroupSizeAttr(parser, key);
    if (failed(parsed))
      return failure();
    subgroupSize = *parsed;
    return success();
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown !hc.group parameter `" << key << "`";
  return failure();
}

Type GroupType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  SMLoc typeLoc = parser.getCurrentLocation();
  ShapeAttr workShape, groupShape;
  ExprAttr subgroupSize;
  if (failed(parser.parseOptionalLess()))
    return GroupType::get(ctx, workShape, groupShape, subgroupSize);
  if (failed(parseAngleBracketedFields(parser, [&](StringRef key) {
        return parseGroupField(parser, key, workShape, groupShape,
                               subgroupSize);
      })))
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

// group_shape / subgroup_size only — no work_shape on nested contexts.
static LogicalResult parseLaunchContextField(AsmParser &parser, StringRef key,
                                             ShapeAttr &groupShape,
                                             ExprAttr &subgroupSize) {
  if (key == "group_shape") {
    FailureOr<ShapeAttr> parsed =
        parseTypedAttr<ShapeAttr>(parser, key, "#hc.shape");
    if (failed(parsed))
      return failure();
    groupShape = *parsed;
    return success();
  }
  if (key == "subgroup_size") {
    FailureOr<ExprAttr> parsed = parseSubgroupSizeAttr(parser, key);
    if (failed(parsed))
      return failure();
    subgroupSize = *parsed;
    return success();
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown launch-context parameter `" << key << "`";
  return failure();
}

template <typename TypeT>
static Type parseNestedLaunchContextType(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  SMLoc typeLoc = parser.getCurrentLocation();
  ShapeAttr groupShape;
  ExprAttr subgroupSize;
  if (failed(parser.parseOptionalLess()))
    return TypeT::get(ctx, groupShape, subgroupSize);
  if (failed(parseAngleBracketedFields(parser, [&](StringRef key) {
        return parseLaunchContextField(parser, key, groupShape, subgroupSize);
      })))
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
