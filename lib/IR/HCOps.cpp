// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::hc;

#define GET_OP_CLASSES
#include "hc/IR/HCOps.cpp.inc"

// `!hc.undef` is the refinement lattice's bottom element: it is compatible
// with any refined type by construction, so verifiers that tighten over the
// iter_args / yield / result triples must treat it as a wildcard to allow
// pre-inference IR to round-trip.
static bool isUndef(Type t) { return llvm::isa<UndefType>(t); }

// Guarded terminator accessor for verifiers. An otherwise-invalid IR (e.g.
// a round-trip bug or a bad builder) could leave a non-empty block with no
// terminator at all; `Block::getTerminator()` asserts in that case, so we
// guard on `mightHaveTerminator()` first and return null for the verifier
// to turn into a diagnostic.
static Operation *tryGetTerminator(Block &block) {
  if (block.empty() || !block.mightHaveTerminator())
    return nullptr;
  return block.getTerminator();
}

LogicalResult HCSymbolOp::verify() {
  // Auto-generated type constraint enforces `!hc.idx` already; all that's
  // left is the "must pin an expression" rule — `!hc.idx` without an
  // expression is the inferred form of an unbound capture, not a
  // user-declared symbol binding.
  if (!llvm::cast<IdxType>(getResult().getType()).getExpr())
    return emitOpError("result must pin a symbolic expression "
                       "(e.g. `!hc.idx<\"M\">`)");
  return success();
}

LogicalResult HCForRangeOp::verify() {
  // The body block takes the induction variable plus one argument per
  // iter_args entry; result types mirror iter_inits types one-to-one.
  Block &body = getBody().front();
  unsigned expectedArgs = 1 + getIterInits().size();
  if (body.getNumArguments() != expectedArgs)
    return emitOpError("expected body block to take ")
           << expectedArgs << " arguments (induction variable + "
           << getIterInits().size() << " iter_args), got "
           << body.getNumArguments();
  if (getIterResults().size() != getIterInits().size())
    return emitOpError("iter_args (")
           << getIterInits().size() << ") and result count ("
           << getIterResults().size() << ") differ";
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip_equal(getIterInits(), getIterResults()))) {
    auto [init, result] = pair;
    if (init.getType() != result.getType())
      return emitOpError("iter_args[")
             << idx << "] type " << init.getType() << " does not match result["
             << idx << "] type " << result.getType();
  }

  // Block arguments must line up with iter_inits one-to-one past the
  // induction variable. `!hc.undef` freely matches anything so pre-inference
  // IR round-trips; once types refine, drift between iter args and block
  // args fails verification.
  for (auto [idx, pair] : llvm::enumerate(llvm::zip_equal(
           getIterInits(), body.getArguments().drop_front(1)))) {
    auto [init, blockArg] = pair;
    Type initTy = init.getType();
    Type blockTy = blockArg.getType();
    if (initTy == blockTy || isUndef(initTy) || isUndef(blockTy))
      continue;
    return emitOpError("iter_args[")
           << idx << "] type " << initTy
           << " does not match body block argument type " << blockTy;
  }

  // The body must terminate with `hc.yield`, and its operands must match
  // the iter_results signature; mirrors `hc.if`. `!hc.undef` is accepted
  // on either side so pre-inference IR round-trips cleanly.
  auto yield = llvm::dyn_cast_or_null<HCYieldOp>(tryGetTerminator(body));
  if (!yield)
    return emitOpError("body must terminate with an `hc.yield`");
  if (yield.getValues().size() != getIterResults().size())
    return emitOpError("body yield produces ")
           << yield.getValues().size() << " values, expected "
           << getIterResults().size();
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip_equal(yield.getValues(), getIterResults()))) {
    auto [yielded, result] = pair;
    Type yieldedTy = yielded.getType();
    Type resultTy = result.getType();
    if (yieldedTy == resultTy || isUndef(yieldedTy) || isUndef(resultTy))
      continue;
    return emitOpError("body yield[")
           << idx << "] type " << yieldedTy << " does not match result[" << idx
           << "] type " << resultTy;
  }
  return success();
}

LogicalResult HCIfOp::verify() {
  // The yield in each non-empty region must produce values matching the op's
  // result types. `!hc.undef` on either side is accepted so pre-inference IR
  // round-trips: `hc.if` is explicitly usable before yield operands have
  // concrete types, and the frontend lowering emits `!hc.undef` on one
  // branch even when the other has refined. An empty else region is fine
  // when the op produces no results, mirroring `scf.if`.
  auto checkRegion = [&](Region &region,
                         llvm::StringRef label) -> LogicalResult {
    if (region.empty())
      return success();
    // `dyn_cast_or_null` + `tryGetTerminator`: a malformed region from a
    // round-trip bug could leave a foreign terminator or no terminator at
    // all; we want a clean verifier error instead of a crash.
    auto yield =
        llvm::dyn_cast_or_null<HCYieldOp>(tryGetTerminator(region.front()));
    if (!yield)
      return emitOpError(label) << " region must terminate with an `hc.yield`";
    if (yield.getValues().size() != getNumResults())
      return emitOpError(label)
             << " yield produces " << yield.getValues().size() << " values, "
             << "expected " << getNumResults();
    for (auto [idx, pair] :
         llvm::enumerate(llvm::zip_equal(yield.getValues(), getResults()))) {
      auto [yielded, result] = pair;
      Type yieldedTy = yielded.getType();
      Type resultTy = result.getType();
      if (yieldedTy == resultTy || isUndef(yieldedTy) || isUndef(resultTy))
        continue;
      return emitOpError(label)
             << " yield[" << idx << "] type " << yieldedTy
             << " does not match result[" << idx << "] type " << resultTy;
    }
    return success();
  };
  if (failed(checkRegion(getThenRegion(), "then")))
    return failure();
  if (failed(checkRegion(getElseRegion(), "else")))
    return failure();
  if (getElseRegion().empty() && getNumResults() != 0)
    return emitOpError("must provide an `else` region when producing results");
  return success();
}

namespace {
/// Extract a concrete shape attribute from a shaped `hc` type, or `nullptr`
/// if the type does not (yet) carry one. Pre-inference IR is typically
/// `!hc.undef`, in which case rank is unknown and axis range cannot be
/// checked — later inference refines the type and picks up the check.
///
/// Fully-qualified `mlir::hc::{Buffer,Tensor,Vector}Type` are required to
/// avoid colliding with MLIR's builtin `VectorType`/`TensorType`.
ShapeAttr tryGetShape(Type t) {
  if (auto buf = llvm::dyn_cast<mlir::hc::BufferType>(t))
    return buf.getShape();
  if (auto tens = llvm::dyn_cast<mlir::hc::TensorType>(t))
    return tens.getShape();
  if (auto vec = llvm::dyn_cast<mlir::hc::VectorType>(t))
    return vec.getShape();
  return nullptr;
}

/// Walks up the parent chain and returns the first subgroup/workitem region
/// op found, or null if the op sits in the default workgroup scope. Stops
/// at the nearest `hc.kernel` / `hc.func` / `hc.intrinsic` / module-like op
/// because nested kernels/funcs re-baseline the enclosing scope.
Operation *findNarrowingScope(Operation *op) {
  Operation *cur = op->getParentOp();
  while (cur) {
    if (llvm::isa<HCSubgroupRegionOp, HCWorkitemRegionOp>(cur))
      return cur;
    if (llvm::isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(cur))
      return nullptr;
    if (cur->hasTrait<OpTrait::SymbolTable>())
      return nullptr;
    cur = cur->getParentOp();
  }
  return nullptr;
}

/// Tensor allocators (`hc.zeros`/`ones`/`full`/`empty`) are workgroup-only —
/// a tensor inside a subgroup or workitem region is a scope error, not a
/// shape error, and calling it out at verify time keeps the diagnostic
/// close to the source instead of surfacing deep in a lowering.
LogicalResult verifyTensorAllocScope(Operation *op) {
  if (Operation *narrowing = findNarrowingScope(op))
    return op->emitOpError("tensor allocator is workgroup scope only; "
                           "enclosed by ")
           << narrowing->getName() << " which narrows the scope";
  return success();
}

} // namespace

LogicalResult HCBufferDimOp::verify() {
  // Python/NumPy semantics allow negative axis indexing, but that is a
  // frontend-time convenience; the dialect form is always canonicalized
  // to a non-negative axis before landing in `hc`. The attr is signless so
  // we read the raw integer value through the stored attribute and check
  // the sign explicitly.
  if (getAxisAttr().getValue().isNegative())
    return emitOpError("axis must be non-negative, got ")
           << getAxisAttr().getValue().getSExtValue();
  // Range check requires concrete rank. Pre-inference `!hc.undef` buffers
  // have no shape metadata, so the check simply skips them; once inference
  // pins the buffer to `!hc.buffer<elem, #hc.shape<...>>` the axis has to
  // fit.
  if (auto shape = tryGetShape(getBuffer().getType())) {
    uint64_t axis = getAxisAttr().getValue().getZExtValue();
    size_t rank = shape.getDims().size();
    if (axis >= rank)
      return emitOpError("axis ")
             << axis << " is out of bounds for rank-" << rank << " buffer";
  }
  return success();
}

namespace {
/// Verify that an optional `#hc.shape` on a load/vload op matches the
/// number of index operands one-to-one. Without a shape attr, we cannot
/// check rank — leave that to inference-stage checks.
template <typename OpT>
LogicalResult verifyLoadShapeRank(OpT op,
                                  mlir::Operation::operand_range indices) {
  auto shape = op.getShapeAttr();
  if (!shape)
    return success();
  size_t expected = shape.getDims().size();
  size_t actual = indices.size();
  if (expected != actual)
    return op.emitOpError("shape rank (")
           << expected << ") does not match index count (" << actual << ")";
  return success();
}
} // namespace

LogicalResult HCLoadOp::verify() {
  return verifyLoadShapeRank(*this, getIndices());
}

LogicalResult HCVLoadOp::verify() {
  return verifyLoadShapeRank(*this, getIndices());
}

LogicalResult HCReduceOp::verify() {
  // Kind is a typed enum now; wrong spellings never reach the verifier.
  if (getAxisAttr().getValue().isNegative())
    return emitOpError("axis must be non-negative, got ")
           << getAxisAttr().getValue().getSExtValue();
  // Same rank-concrete-only story as `hc.buffer_dim`: when inference has
  // pinned the value to a shaped `hc` type, reject out-of-range axes;
  // otherwise defer to inference.
  if (auto shape = tryGetShape(getValue().getType())) {
    uint64_t axis = getAxisAttr().getValue().getZExtValue();
    size_t rank = shape.getDims().size();
    if (axis >= rank)
      return emitOpError("axis ")
             << axis << " is out of bounds for rank-" << rank << " value";
  }
  return success();
}

LogicalResult HCAsTypeOp::verify() {
  // `hc.astype` models numeric conversion and nothing else. Anything that is
  // not a builtin numeric scalar (int/float/index) is rejected so
  // `target = !hc.slice` and similar nonsense fail at verify.
  Type target = getTarget();
  if (!target.isIntOrIndexOrFloat())
    return emitOpError("target type must be a builtin integer, index, or "
                       "float type, got ")
           << target;
  // The op's declared result type must agree with `target`: scalar results
  // match it directly; tensor/vector results agree on element type; the
  // `!hc.undef` escape keeps pre-inference IR legal. Anything else is a
  // builder bug that should fail loudly instead of silently round-tripping.
  Type result = getResult().getType();
  if (isUndef(result))
    return success();
  if (result.isIntOrIndexOrFloat()) {
    if (result != target)
      return emitOpError("result type ")
             << result << " does not match target type " << target;
    return success();
  }
  auto elementOf = [](Type t) -> Type {
    if (auto tens = llvm::dyn_cast<mlir::hc::TensorType>(t))
      return tens.getElementType();
    if (auto vec = llvm::dyn_cast<mlir::hc::VectorType>(t))
      return vec.getElementType();
    return {};
  };
  if (Type elem = elementOf(result)) {
    if (elem != target)
      return emitOpError("result element type ")
             << elem << " does not match target type " << target;
    return success();
  }
  return emitOpError("result type ")
         << result
         << " must be `!hc.undef`, a builtin numeric scalar, or an "
            "`!hc.tensor`/`!hc.vector` whose element type matches target";
}

LogicalResult HCWithInactiveOp::verify() {
  // `$inactive` is a scalar literal attribute; its numeric kind must agree
  // with the masked value's element domain. The check runs only once
  // inference pins the operand to a shaped type with an element; pre-
  // inference `!hc.undef` operands skip straight to success so the op
  // remains usable out of the mechanical frontend pass.
  Type value = getValue().getType();
  if (isUndef(value) || value.isIntOrIndexOrFloat())
    return success();
  Type elem;
  if (auto tens = llvm::dyn_cast<mlir::hc::TensorType>(value))
    elem = tens.getElementType();
  else if (auto vec = llvm::dyn_cast<mlir::hc::VectorType>(value))
    elem = vec.getElementType();
  if (!elem || isUndef(elem))
    return success();
  Attribute inactive = getInactiveAttr();
  auto sameDomain = [&](Attribute a) -> bool {
    if (llvm::isa<BoolAttr>(a))
      return elem.isInteger(1);
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(a))
      return elem == intAttr.getType();
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(a))
      return elem == floatAttr.getType();
    return false;
  };
  if (!sameDomain(inactive))
    return emitOpError("inactive literal ")
           << inactive << " does not match element type " << elem;
  return success();
}

LogicalResult HCZerosOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCOnesOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCFullOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCEmptyOp::verify() { return verifyTensorAllocScope(*this); }

//===----------------------------------------------------------------------===//
// SymbolUserOpInterface verification for call ops.
//
// Callee existence and op kind are cheap; signature parity is also verified
// when the callee carries a `function_type` attribute. `!hc.undef` on either
// side of the parity check passes (progressive typing policy): a call site
// that has not yet been inferred, or a signature that still lists `!hc.undef`
// placeholders, should not cause spurious verify errors.
//===----------------------------------------------------------------------===//

namespace {
// Signature compatibility with progressive-typing escape hatch.
bool compatibleSigType(Type callSite, Type callee) {
  if (isUndef(callSite) || isUndef(callee))
    return true;
  return callSite == callee;
}

template <typename CallOp>
LogicalResult verifySignature(CallOp op, FunctionType fnType) {
  if (fnType.getNumInputs() != op.getArgs().size())
    return op.emitOpError("callee '@")
           << op.getCallee() << "' expects " << fnType.getNumInputs()
           << " argument(s), call site provides " << op.getArgs().size();
  if (fnType.getNumResults() != op.getResults().size())
    return op.emitOpError("callee '@")
           << op.getCallee() << "' returns " << fnType.getNumResults()
           << " result(s), call site declares " << op.getResults().size();
  for (auto [i, callSite, declared] :
       llvm::enumerate(op.getArgs().getTypes(), fnType.getInputs())) {
    if (!compatibleSigType(callSite, declared))
      return op.emitOpError("arg #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  for (auto [i, callSite, declared] :
       llvm::enumerate(op.getResults().getTypes(), fnType.getResults())) {
    if (!compatibleSigType(callSite, declared))
      return op.emitOpError("result #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  return success();
}

template <typename CalleeOp, typename CallOp>
LogicalResult verifyFlatSymbolUseAsOp(CallOp op,
                                      SymbolTableCollection &symbolTable,
                                      llvm::StringRef expectedKindLabel) {
  auto sym = symbolTable.lookupNearestSymbolFrom<CalleeOp>(op.getOperation(),
                                                           op.getCalleeAttr());
  if (!sym)
    return op.emitOpError("'")
           << op.getCallee() << "' does not reference a valid "
           << expectedKindLabel;
  if (std::optional<FunctionType> fnType = sym.getFunctionType())
    return verifySignature(op, *fnType);
  return success();
}
} // namespace

LogicalResult HCCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyFlatSymbolUseAsOp<HCFuncOp>(*this, symbolTable, "hc.func");
}

namespace {
// Translates the declared effect class into concrete side effects on the
// default resource. The callee's body is opaque; we only know "maybe reads"
// / "maybe writes" at this level, so `Pure` emits nothing, the one-sided
// classes emit the matching effect, and the unknown/absent case falls back
// to MemRead+MemWrite.
void emitEffectsForClass(
    std::optional<EffectClass> cls,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto value = cls.value_or(EffectClass::ReadWrite);
  if (value == EffectClass::Pure)
    return;
  if (value == EffectClass::Read || value == EffectClass::ReadWrite)
    effects.emplace_back(MemoryEffects::Read::get());
  if (value == EffectClass::Write || value == EffectClass::ReadWrite)
    effects.emplace_back(MemoryEffects::Write::get());
}

template <typename CalleeOp, typename CallOp>
void populateEffectsFromCallee(
    CallOp op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  std::optional<EffectClass> cls;
  if (auto mod = op->template getParentOfType<ModuleOp>()) {
    if (auto callee = mod.template lookupSymbol<CalleeOp>(op.getCalleeAttr()))
      cls = callee.getEffects();
  }
  emitEffectsForClass(cls, effects);
}
} // namespace

void HCCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  populateEffectsFromCallee<HCFuncOp>(*this, effects);
}

LogicalResult
HCCallIntrinsicOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyFlatSymbolUseAsOp<HCIntrinsicOp>(*this, symbolTable,
                                                    "hc.intrinsic")))
    return failure();
  // Callee exists and has the right kind; now enforce the const_kwarg
  // whitelist it declared, if any. Extra attributes on the call site are
  // allowed (forward-compatible with target-specific decorations).
  auto intrinsic = symbolTable.lookupNearestSymbolFrom<HCIntrinsicOp>(
      getOperation(), getCalleeAttr());
  ArrayAttr required = intrinsic.getConstKwargsAttr();
  if (!required)
    return success();
  for (Attribute entry : required) {
    llvm::StringRef name = llvm::cast<StringAttr>(entry).getValue();
    if (!(*this)->hasAttr(name))
      return emitOpError("missing required const kwarg '")
             << name << "' declared by callee '@" << getCallee() << "'";
  }
  return success();
}

void HCCallIntrinsicOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  populateEffectsFromCallee<HCIntrinsicOp>(*this, effects);
}
