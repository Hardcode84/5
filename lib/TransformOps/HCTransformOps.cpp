// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/TransformOps/HCTransformOps.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hc::transform;
namespace xform = mlir::transform;

template <typename TransformOp>
static DiagnosedSilenceableFailure
requireSinglePayloadOp(TransformOp transformOp, Value handle,
                       xform::TransformState &state, const Twine &role,
                       Operation *&payloadOp) {
  SmallVector<Operation *> payloadOps =
      llvm::to_vector(state.getPayloadOps(handle));
  if (payloadOps.size() != 1) {
    return transformOp.emitSilenceableError()
           << "expected exactly one " << role << " payload op, got "
           << payloadOps.size();
  }
  payloadOp = payloadOps.front();
  return DiagnosedSilenceableFailure::success();
}

template <typename TransformOp>
static DiagnosedSilenceableFailure
requireSinglePayloadValue(TransformOp transformOp, Value handle,
                          xform::TransformState &state, const Twine &role,
                          Value &payloadValue) {
  SmallVector<Value> payloadValues =
      llvm::to_vector(state.getPayloadValues(handle));
  if (payloadValues.size() != 1) {
    return transformOp.emitSilenceableError()
           << "expected exactly one " << role << " payload value, got "
           << payloadValues.size();
  }
  payloadValue = payloadValues.front();
  return DiagnosedSilenceableFailure::success();
}

template <typename TransformOp>
static DiagnosedSilenceableFailure
requireSingleParam(TransformOp transformOp, Value handle,
                   xform::TransformState &state, const Twine &role,
                   Attribute &param) {
  ArrayRef<Attribute> params = state.getParams(handle);
  if (params.size() != 1) {
    return transformOp.emitSilenceableError()
           << "expected exactly one " << role << " parameter, got "
           << params.size();
  }
  param = params.front();
  return DiagnosedSilenceableFailure::success();
}

static FailureOr<unsigned> findRuntimeOperandIndex(hc::HCCallIntrinsicOp call,
                                                   StringRef name) {
  auto intrinsic = SymbolTable::lookupNearestSymbolFrom<hc::HCIntrinsicOp>(
      call.getOperation(), call.getCalleeAttr());
  std::optional<ArrayAttr> parameters =
      intrinsic ? intrinsic.getParameters() : std::optional<ArrayAttr>();
  if (!parameters)
    return failure();

  ArrayAttr operands = hc::filterIntrinsicOperandParameters(
      *parameters, intrinsic.getConstKwargsAttr());
  for (auto [index, attr] : llvm::enumerate(operands)) {
    auto stringAttr = dyn_cast<StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return static_cast<unsigned>(index);
  }
  return failure();
}

DiagnosedSilenceableFailure
HCTransformMatchIntrinsicCallOp::apply(xform::TransformRewriter &rewriter,
                                       xform::TransformResults &results,
                                       xform::TransformState &state) {
  SmallVector<Operation *> matched;
  for (Operation *root : state.getPayloadOps(getRoot())) {
    root->walk([&](hc::HCCallIntrinsicOp call) {
      if (call.getCallee() == getCallee())
        matched.push_back(call.getOperation());
    });
  }
  results.set(cast<OpResult>(getMatched()), matched);
  return DiagnosedSilenceableFailure::success();
}

void HCTransformMatchIntrinsicCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getRootMutable(), effects);
  xform::producesHandle(getOperation()->getOpResults(), effects);
  xform::onlyReadsPayload(effects);
}

LogicalResult HCTransformGetIntrinsicOperandOp::verify() {
  bool hasName = static_cast<bool>(getName());
  bool hasIndex = static_cast<bool>(getIndex());
  if (hasName == hasIndex)
    return emitOpError("requires exactly one of name or index");
  if (getIndex() && getIndexAttr().getInt() < 0)
    return emitOpError("requires a non-negative operand index");
  return success();
}

DiagnosedSilenceableFailure
HCTransformGetIntrinsicOperandOp::apply(xform::TransformRewriter &rewriter,
                                        xform::TransformResults &results,
                                        xform::TransformState &state) {
  Operation *payloadOp = nullptr;
  DiagnosedSilenceableFailure diag = requireSinglePayloadOp(
      *this, getCall(), state, "intrinsic call", payloadOp);
  if (!diag.succeeded())
    return diag;
  auto call = dyn_cast<hc::HCCallIntrinsicOp>(payloadOp);
  if (!call) {
    return emitSilenceableError() << "expected an hc.call_intrinsic payload op";
  }

  unsigned index = 0;
  if (getIndex()) {
    index = static_cast<unsigned>(getIndexAttr().getInt());
  } else {
    FailureOr<unsigned> maybeIndex =
        findRuntimeOperandIndex(call, getNameAttr().getValue());
    if (failed(maybeIndex)) {
      return emitSilenceableError()
             << "failed to resolve intrinsic operand " << getNameAttr();
    }
    index = *maybeIndex;
  }

  if (index >= call->getNumOperands()) {
    return emitSilenceableError()
           << "intrinsic operand index " << index << " out of range";
  }
  results.setValues(cast<OpResult>(getValue()), {call->getOperand(index)});
  return DiagnosedSilenceableFailure::success();
}

void HCTransformGetIntrinsicOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getCallMutable(), effects);
  xform::producesHandle(getOperation()->getOpResults(), effects);
  xform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
HCTransformGetIntrinsicResultTypeOp::apply(xform::TransformRewriter &rewriter,
                                           xform::TransformResults &results,
                                           xform::TransformState &state) {
  Operation *payloadOp = nullptr;
  DiagnosedSilenceableFailure diag = requireSinglePayloadOp(
      *this, getCall(), state, "intrinsic call", payloadOp);
  if (!diag.succeeded())
    return diag;
  auto call = dyn_cast<hc::HCCallIntrinsicOp>(payloadOp);
  if (!call) {
    return emitSilenceableError() << "expected an hc.call_intrinsic payload op";
  }
  int64_t index = getIndex();
  if (index < 0 || static_cast<unsigned>(index) >= call->getNumResults()) {
    return emitSilenceableError()
           << "intrinsic result index " << index << " out of range";
  }
  results.setParams(cast<OpResult>(getType()),
                    {TypeAttr::get(call->getResult(index).getType())});
  return DiagnosedSilenceableFailure::success();
}

void HCTransformGetIntrinsicResultTypeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getCallMutable(), effects);
  xform::producesHandle(getOperation()->getOpResults(), effects);
  xform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
HCTransformGetIntrinsicAttrOp::apply(xform::TransformRewriter &rewriter,
                                     xform::TransformResults &results,
                                     xform::TransformState &state) {
  Operation *payloadOp = nullptr;
  DiagnosedSilenceableFailure diag = requireSinglePayloadOp(
      *this, getCall(), state, "intrinsic call", payloadOp);
  if (!diag.succeeded())
    return diag;
  auto call = dyn_cast<hc::HCCallIntrinsicOp>(payloadOp);
  if (!call) {
    return emitSilenceableError() << "expected an hc.call_intrinsic payload op";
  }
  Attribute attr = call->getAttr(getName());
  if (!attr) {
    return emitSilenceableError()
           << "intrinsic call has no attribute " << getName();
  }
  results.setParams(cast<OpResult>(getParam()), {attr});
  return DiagnosedSilenceableFailure::success();
}

void HCTransformGetIntrinsicAttrOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getCallMutable(), effects);
  xform::producesHandle(getOperation()->getOpResults(), effects);
  xform::onlyReadsPayload(effects);
}

LogicalResult HCTransformCreateOp::verify() {
  if (getDynamicAttrNames().size() != getDynamicAttrs().size()) {
    return emitOpError()
           << "requires dynamic_attr_names to match dynamic_attrs arity";
  }
  return success();
}

// Materialise the runtime operand list from the transform op's input
// handles: each handle must resolve to exactly one payload value.
static DiagnosedSilenceableFailure
collectPayloadOperands(HCTransformCreateOp self, xform::TransformState &state,
                       SmallVectorImpl<Value> &out) {
  for (auto [index, operand] : llvm::enumerate(self.getInputs())) {
    Value payloadValue;
    DiagnosedSilenceableFailure diag = requireSinglePayloadValue(
        self, operand, state, Twine("operand ") + Twine(index), payloadValue);
    if (!diag.succeeded())
      return diag;
    out.push_back(payloadValue);
  }
  return DiagnosedSilenceableFailure::success();
}

// Materialise the payload op's result types from `result_types` handles.
// Each handle must resolve to exactly one parameter which itself must be
// a TypeAttr.
static DiagnosedSilenceableFailure
collectPayloadResultTypes(HCTransformCreateOp self,
                          xform::TransformState &state,
                          SmallVectorImpl<Type> &out) {
  for (auto [index, typeParam] : llvm::enumerate(self.getResultTypes())) {
    Attribute attr;
    DiagnosedSilenceableFailure diag = requireSingleParam(
        self, typeParam, state, Twine("result type ") + Twine(index), attr);
    if (!diag.succeeded())
      return diag;
    auto typeAttr = dyn_cast<TypeAttr>(attr);
    if (!typeAttr)
      return self.emitSilenceableError()
             << "result type parameter " << index << " is not a TypeAttr";
    out.push_back(typeAttr.getValue());
  }
  return DiagnosedSilenceableFailure::success();
}

// Materialise the payload op's attribute dictionary by concatenating the
// static_attrs dict (if any) with the dynamic_attrs resolved from their
// handles.
static DiagnosedSilenceableFailure collectPayloadAttributes(
    HCTransformCreateOp self, xform::TransformRewriter &rewriter,
    xform::TransformState &state, SmallVectorImpl<NamedAttribute> &out) {
  if (std::optional<DictionaryAttr> staticAttrs = self.getStaticAttrs())
    llvm::append_range(out, staticAttrs->getValue());
  for (auto [nameAttr, attrParam] :
       llvm::zip_equal(self.getDynamicAttrNames(), self.getDynamicAttrs())) {
    Attribute attr;
    DiagnosedSilenceableFailure diag =
        requireSingleParam(self, attrParam, state,
                           Twine("dynamic attribute ") +
                               Twine(cast<StringAttr>(nameAttr).getValue()),
                           attr);
    if (!diag.succeeded())
      return diag;
    out.push_back(
        rewriter.getNamedAttr(cast<StringAttr>(nameAttr).getValue(), attr));
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
HCTransformCreateOp::apply(xform::TransformRewriter &rewriter,
                           xform::TransformResults &results,
                           xform::TransformState &state) {
  Operation *insertionPoint = nullptr;
  DiagnosedSilenceableFailure diag = requireSinglePayloadOp(
      *this, getInsertionPoint(), state, "insertion point", insertionPoint);
  if (!diag.succeeded())
    return diag;

  SmallVector<Value> payloadOperands;
  diag = collectPayloadOperands(*this, state, payloadOperands);
  if (!diag.succeeded())
    return diag;

  SmallVector<Type> payloadResultTypes;
  diag = collectPayloadResultTypes(*this, state, payloadResultTypes);
  if (!diag.succeeded())
    return diag;

  SmallVector<NamedAttribute> payloadAttrs;
  diag = collectPayloadAttributes(*this, rewriter, state, payloadAttrs);
  if (!diag.succeeded())
    return diag;

  OperationState opState(insertionPoint->getLoc(), getOpName());
  opState.addOperands(payloadOperands);
  opState.addTypes(payloadResultTypes);
  opState.addAttributes(payloadAttrs);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertionPoint);
  Operation *created = rewriter.create(opState);
  if (created->getNumResults() != getOperation()->getNumResults()) {
    return DiagnosedSilenceableFailure::definiteFailure()
           << "created payload op produced " << created->getNumResults()
           << " results but transform op has "
           << getOperation()->getNumResults();
  }
  for (auto [transformResult, payloadResult] :
       llvm::zip_equal(getOperation()->getResults(), created->getResults())) {
    results.setValues(cast<OpResult>(transformResult), {payloadResult});
  }
  return DiagnosedSilenceableFailure::success();
}

void HCTransformCreateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getOperation()->getOpOperands(), effects);
  xform::producesHandle(getOperation()->getOpResults(), effects);
  xform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
HCTransformConstantTypeOp::apply(xform::TransformRewriter &rewriter,
                                 xform::TransformResults &results,
                                 xform::TransformState &state) {
  results.setParams(cast<OpResult>(getResult()), {TypeAttr::get(getValue())});
  return DiagnosedSilenceableFailure::success();
}

void HCTransformConstantTypeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // The op is `Pure` at the TableGen level (no side effects); only the
  // result handle needs an explicit producer effect so the transform
  // verifier can wire downstream uses.
  xform::producesHandle(getOperation()->getOpResults(), effects);
}

DiagnosedSilenceableFailure
HCTransformCastValueOp::apply(xform::TransformRewriter &rewriter,
                              xform::TransformResults &results,
                              xform::TransformState &state) {
  Value source;
  DiagnosedSilenceableFailure diag =
      requireSinglePayloadValue(*this, getSource(), state, "source", source);
  if (!diag.succeeded())
    return diag;

  Attribute targetAttr;
  diag = requireSingleParam(*this, getTargetType(), state, "target type",
                            targetAttr);
  if (!diag.succeeded())
    return diag;
  auto targetTypeAttr = dyn_cast<TypeAttr>(targetAttr);
  if (!targetTypeAttr) {
    return emitSilenceableError() << "target_type parameter is not a TypeAttr";
  }
  Type targetType = targetTypeAttr.getValue();

  Value resultValue = source;
  if (source.getType() != targetType) {
    // Anchor the cast at the source's definition site so any future
    // create_op consuming this handle sees a dominator that is independent
    // of the consumer's eventual placement. Block-argument sources land
    // their cast at the start of the owning block.
    OpBuilder::InsertionGuard guard(rewriter);
    if (Operation *defOp = source.getDefiningOp()) {
      rewriter.setInsertionPointAfter(defOp);
    } else {
      auto blockArg = cast<BlockArgument>(source);
      rewriter.setInsertionPointToStart(blockArg.getOwner());
    }
    auto cast = UnrealizedConversionCastOp::create(
        rewriter, source.getLoc(), TypeRange{targetType}, ValueRange{source});
    resultValue = cast.getResult(0);
  }
  results.setValues(::mlir::cast<OpResult>(getResult()), {resultValue});
  return DiagnosedSilenceableFailure::success();
}

void HCTransformCastValueOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getSourceMutable(), effects);
  xform::onlyReadsHandle(getTargetTypeMutable(), effects);
  xform::producesHandle(getOperation()->getOpResults(), effects);
  // The cast is a real payload mutation when source/target types differ;
  // the no-op forwarding case still benefits from declaring `modifiesPayload`
  // because canonicalize would otherwise be tempted to eliminate the
  // op-level side-effect-free producer-of-only-reads-handle pair.
  xform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
HCTransformReplaceIntrinsicCallOp::apply(xform::TransformRewriter &rewriter,
                                         xform::TransformResults &results,
                                         xform::TransformState &state) {
  Operation *payloadOp = nullptr;
  DiagnosedSilenceableFailure diag = requireSinglePayloadOp(
      *this, getCall(), state, "intrinsic call", payloadOp);
  if (!diag.succeeded())
    return diag;
  auto call = dyn_cast<hc::HCCallIntrinsicOp>(payloadOp);
  if (!call) {
    return emitSilenceableError() << "expected an hc.call_intrinsic payload op";
  }

  SmallVector<Value> replacementValues;
  for (auto [index, replacement] : llvm::enumerate(getReplacements())) {
    Value payloadValue;
    diag = requireSinglePayloadValue(*this, replacement, state,
                                     Twine("replacement ") + Twine(index),
                                     payloadValue);
    if (!diag.succeeded())
      return diag;
    replacementValues.push_back(payloadValue);
  }
  if (replacementValues.size() != call->getNumResults()) {
    return emitSilenceableError()
           << "replacement count " << replacementValues.size()
           << " does not match intrinsic result count "
           << call->getNumResults();
  }
  rewriter.replaceOp(call, replacementValues);
  return DiagnosedSilenceableFailure::success();
}

void HCTransformReplaceIntrinsicCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::consumesHandle(getCallMutable(), effects);
  xform::onlyReadsHandle(getReplacementsMutable(), effects);
  xform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
HCTransformRequireIntrinsicAttrOp::apply(xform::TransformRewriter &rewriter,
                                         xform::TransformResults &results,
                                         xform::TransformState &state) {
  Operation *payloadOp = nullptr;
  DiagnosedSilenceableFailure diag = requireSinglePayloadOp(
      *this, getCall(), state, "intrinsic call", payloadOp);
  if (!diag.succeeded())
    return diag;
  auto call = dyn_cast<hc::HCCallIntrinsicOp>(payloadOp);
  if (!call)
    return emitSilenceableError() << "expected an hc.call_intrinsic payload op";

  StringRef name = getName();
  Attribute expected = getExpected();
  Attribute actual = call->getAttr(name);
  if (!actual) {
    auto fail = emitDefiniteFailure();
    fail << "intrinsic call @" << call.getCallee()
         << " missing required attribute '" << name << "'";
    fail.attachNote(call.getLoc()) << "call site";
    return fail;
  }
  if (actual != expected) {
    auto fail = emitDefiniteFailure();
    fail << "intrinsic call @" << call.getCallee() << " has " << name << " = "
         << actual << ", expected " << expected;
    fail.attachNote(call.getLoc()) << "call site";
    return fail;
  }
  return DiagnosedSilenceableFailure::success();
}

void HCTransformRequireIntrinsicAttrOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  xform::onlyReadsHandle(getCallMutable(), effects);
  // Conceptually this op only reads the payload — failure isn't a memory
  // effect — but `onlyReadsPayload` lets the canonicalizer treat the op as
  // dead when it has no SSA users (and a require op never does). Declaring
  // payload as written keeps the op alive past `--canonicalize`/`--cse`,
  // which the recipe IR has to survive so the interpreter can still see
  // the assertion.
  xform::modifiesPayload(effects);
}

namespace {
class HCTransformDialectExtension
    : public xform::TransformDialectExtension<HCTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCTransformDialectExtension)

  HCTransformDialectExtension() {
    declareGeneratedDialect<hc::HCDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "hc/TransformOps/HCTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "hc/TransformOps/HCTransformOps.cpp.inc"

void mlir::hc::transform::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<HCTransformDialectExtension>();
}
