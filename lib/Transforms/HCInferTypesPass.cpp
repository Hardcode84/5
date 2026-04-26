// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-infer-types`, an opt-in pass that refines the placeholder
// `!hc.undef` type left by mechanical frontend lowering. The dataflow analysis
// computes facts to a solver fixpoint; a separate monotonic rewrite step then
// updates IR result/block-argument types. If that rewrite exposes more concrete
// surface types to later transfer functions, the pass reruns the solver until
// the type surface stops changing.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir::hc {
#define GEN_PASS_DEF_HCINFERTYPES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

constexpr llvm::StringLiteral kSyntheticJoinPrefix = "$join";
constexpr llvm::StringLiteral kSyntheticJoinTmpPrefix = "$__hc_join_tmp_";

static bool consumeDigits(StringRef &text) {
  size_t end = 0;
  while (end < text.size() && text[end] >= '0' && text[end] <= '9')
    ++end;
  if (end == 0)
    return false;
  text = text.drop_front(end);
  return true;
}

static bool isSyntheticJoinSymbolName(StringRef name) {
  StringRef rest = name;
  if (rest.consume_front(kSyntheticJoinPrefix))
    return consumeDigits(rest) && rest.empty();

  if (!rest.consume_front(kSyntheticJoinTmpPrefix) || !consumeDigits(rest))
    return false;
  while (!rest.empty()) {
    if (!rest.consume_front("_") || !consumeDigits(rest))
      return false;
  }
  return true;
}

struct TypeFact {
  enum class Kind { Unknown, Concrete, Conflict };

  Kind kind = Kind::Unknown;
  Type type;

  static TypeFact unknown() { return {}; }
  static TypeFact concrete(Type type) {
    TypeFact fact;
    fact.kind = Kind::Concrete;
    fact.type = type;
    return fact;
  }
  static TypeFact conflict() {
    TypeFact fact;
    fact.kind = Kind::Conflict;
    return fact;
  }
  static TypeFact conflict(Type type) {
    TypeFact fact;
    fact.kind = Kind::Conflict;
    fact.type = type;
    return fact;
  }

  bool isUnknown() const { return kind == Kind::Unknown; }
  bool isConcrete() const { return kind == Kind::Concrete && type; }
  bool isConflict() const { return kind == Kind::Conflict; }
  bool isHardConflict() const { return isConflict() && !type; }
  bool hasUsableType() const { return isConcrete() || (isConflict() && type); }

  bool operator==(const TypeFact &rhs) const {
    return kind == rhs.kind && type == rhs.type;
  }

  static TypeFact join(const TypeFact &lhs, const TypeFact &rhs) {
    if (lhs.isUnknown())
      return rhs;
    if (rhs.isUnknown())
      return lhs;
    if (lhs.isConflict())
      return lhs;
    if (rhs.isConflict())
      return rhs;
    if (Type common = joinHCTypes(lhs.type, rhs.type))
      return concrete(common);
    return conflict();
  }

  void print(raw_ostream &os) const {
    // `dataflow::Lattice` calls this when dumping analysis state.
    if (isUnknown()) {
      os << "<unknown>";
      return;
    }
    if (isConflict()) {
      os << "<conflict";
      if (type)
        os << ": " << type;
      os << ">";
      return;
    }
    os << type;
  }
};

static bool containsSyntheticJoinSymbol(IdxType type);

template <typename JoinIdxConflictFn>
static Type joinConcreteTypesWithPolicy(Type lhs, Type rhs,
                                        SmallVectorImpl<unsigned> &elementPath,
                                        JoinIdxConflictFn joinIdxConflict) {
  if (lhs == rhs || isHCUndefType(lhs))
    return rhs;
  if (isHCUndefType(rhs))
    return lhs;

  auto lhsTuple = dyn_cast<TupleType>(lhs);
  auto rhsTuple = dyn_cast<TupleType>(rhs);
  if (lhsTuple || rhsTuple) {
    if (!lhsTuple || !rhsTuple || lhsTuple.size() != rhsTuple.size())
      return {};
    SmallVector<Type> elements;
    elements.reserve(lhsTuple.size());
    unsigned elementIndex = 0;
    for (auto [lhsElement, rhsElement] :
         llvm::zip_equal(lhsTuple.getTypes(), rhsTuple.getTypes())) {
      elementPath.push_back(elementIndex++);
      Type joined = joinConcreteTypesWithPolicy(lhsElement, rhsElement,
                                                elementPath, joinIdxConflict);
      elementPath.pop_back();
      if (!joined)
        return {};
      elements.push_back(joined);
    }
    return TupleType::get(lhs.getContext(), elements);
  }

  auto lhsIdx = dyn_cast<IdxType>(lhs);
  auto rhsIdx = dyn_cast<IdxType>(rhs);
  if (lhsIdx && rhsIdx && lhsIdx.getExpr() && rhsIdx.getExpr()) {
    // Once a conflict has a synthetic representative, keep it stable across
    // later solver reruns and expressions derived from that representative.
    if (containsSyntheticJoinSymbol(lhsIdx))
      return lhs;
    if (containsSyntheticJoinSymbol(rhsIdx))
      return rhs;
    if (Type joined = joinIdxConflict(lhs.getContext(), elementPath))
      return joined;
  }

  return joinHCTypes(lhs, rhs);
}

struct HCTypeLattice : public dataflow::Lattice<TypeFact> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCTypeLattice)
  using Lattice::Lattice;

  ChangeResult join(const dataflow::AbstractSparseLattice &rhs) override {
    return join(static_cast<const HCTypeLattice &>(rhs).getValue());
  }

  ChangeResult join(const TypeFact &rhs) {
    TypeFact joined = joinFacts(getValue(), rhs);
    if (joined == getValue())
      return ChangeResult::NoChange;
    getValue() = joined;
    return ChangeResult::Change;
  }

  ChangeResult joinSyntheticIdxRepresentative(MLIRContext *ctx) {
    return join(TypeFact::conflict(synthesizeIdxConflictType(ctx)));
  }

private:
  Type synthesizeIdxConflictType(MLIRContext *ctx,
                                 ArrayRef<unsigned> elementPath = {}) const {
    SmallString<48> text(kSyntheticJoinTmpPrefix);
    text += Twine(reinterpret_cast<uintptr_t>(this)).str();
    for (unsigned element : elementPath) {
      text += "_";
      text += Twine(element).str();
    }
    auto *dialect = ctx->getOrLoadDialect<HCDialect>();
    std::string diag;
    FailureOr<sym::ExprHandle> handle =
        sym::parseExpr(dialect->getSymbolStore(), text, &diag);
    if (failed(handle))
      llvm::report_fatal_error("failed to parse synthesized idx symbol");
    return IdxType::get(ctx, ExprAttr::get(ctx, *handle));
  }

  TypeFact joinFacts(const TypeFact &lhs, const TypeFact &rhs) const {
    if (lhs.isUnknown())
      return rhs;
    if (rhs.isUnknown())
      return lhs;
    if (lhs.isConflict())
      return lhs;
    if (rhs.isConflict())
      return rhs;
    SmallVector<unsigned, 4> elementPath;
    auto synthesizeIdxConflict = [&](MLIRContext *ctx,
                                     ArrayRef<unsigned> path) {
      return synthesizeIdxConflictType(ctx, path);
    };
    if (Type common = joinConcreteTypesWithPolicy(
            lhs.type, rhs.type, elementPath, synthesizeIdxConflict))
      return TypeFact::concrete(common);
    return TypeFact::conflict();
  }
};

static TypeFact factFromExistingType(Type type) {
  if (!type || isHCUndefType(type))
    return TypeFact::unknown();
  return TypeFact::concrete(type);
}

static TypeFact factFromLattice(const HCTypeLattice *lattice) {
  if (!lattice)
    return TypeFact::unknown();
  return lattice->getValue();
}

static bool containsSyntheticJoinSymbol(IdxType type) {
  ExprAttr expr = type.getExpr();
  if (!expr)
    return false;
  bool found = false;
  sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
    found |= isSyntheticJoinSymbolName(name);
  });
  return found;
}

static Type joinExistingConcreteTypes(Type lhs, Type rhs) {
  SmallVector<unsigned, 4> elementPath;
  auto noSyntheticRepresentative =
      [](MLIRContext *, ArrayRef<unsigned>) -> Type { return {}; };
  return joinConcreteTypesWithPolicy(lhs, rhs, elementPath,
                                     noSyntheticRepresentative);
}

static TypeFact joinExistingFacts(const TypeFact &lhs, const TypeFact &rhs) {
  if (lhs.isUnknown())
    return rhs;
  if (rhs.isUnknown())
    return lhs;
  if (lhs.isConflict())
    return lhs;
  if (rhs.isConflict())
    return rhs;
  if (Type common = joinExistingConcreteTypes(lhs.type, rhs.type))
    return TypeFact::concrete(common);
  return TypeFact::conflict();
}

static bool isUnpinnedIdxType(Type type) {
  auto idx = dyn_cast_or_null<IdxType>(type);
  return idx && !idx.getExpr();
}

class HCTypeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<HCTypeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(HCTypeLattice *lattice) override {
    join(lattice, factFromExistingType(lattice->getAnchor().getType()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const HCTypeLattice *> operands,
                               ArrayRef<HCTypeLattice *> results) override {
    for (auto [result, lattice] : llvm::zip(op->getResults(), results))
      join(lattice, factFromExistingType(result.getType()));

    if (auto infer = dyn_cast<HCInferTypeOpInterface>(op)) {
      SmallVector<Type> operandTypes;
      operandTypes.reserve(operands.size());
      for (const HCTypeLattice *operand : operands) {
        TypeFact fact = factFromLattice(operand);
        operandTypes.push_back(fact.hasUsableType() ? fact.type : Type{});
      }

      SmallVector<Type> inferredTypes;
      if (failed(infer.inferHCTypes(operandTypes, inferredTypes)))
        return failure();
      if (inferredTypes.size() != results.size()) {
        op->emitOpError("inferred ")
            << inferredTypes.size() << " result type(s), but op has "
            << results.size() << " result(s)";
        return failure();
      }

      for (auto [lattice, type] : llvm::zip(results, inferredTypes)) {
        if (type)
          join(lattice, TypeFact::concrete(type));
      }
      return success();
    }

    return success();
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<HCTypeLattice *> nonSuccessorInputLattices) override {
    assert(nonSuccessorInputs.size() == nonSuccessorInputLattices.size() &&
           "size mismatch");

    if (auto forRange = dyn_cast<HCForRangeOp>(op)) {
      if (!successor.isParent() && !nonSuccessorInputs.empty()) {
        Type ivType = nonSuccessorInputs.front().getType();
        HCTypeLattice *ivLattice = nonSuccessorInputLattices.front();
        // An unknown or unpinned IV is still a distinct symbolic value; use the
        // same representative machinery as idx joins so IV-derived expressions
        // retain something to reason about.
        if (isHCUndefType(ivType) || isUnpinnedIdxType(ivType))
          propagateIfChanged(
              ivLattice,
              ivLattice->joinSyntheticIdxRepresentative(forRange.getContext()));
        else
          join(ivLattice, factFromExistingType(ivType));
      }
      for (auto [input, lattice] :
           llvm::zip(nonSuccessorInputs.drop_front(),
                     nonSuccessorInputLattices.drop_front()))
        join(lattice, factFromExistingType(input.getType()));
      return;
    }

    auto infer = dyn_cast<HCInferRegionArgTypeOpInterface>(op);
    if (!infer)
      return setAllToEntryStates(nonSuccessorInputLattices);

    SmallVector<Type> inferredTypes;
    if (failed(infer.inferHCRegionArgTypes(successor, nonSuccessorInputs,
                                           inferredTypes)))
      return setAllToEntryStates(nonSuccessorInputLattices);
    if (inferredTypes.size() != nonSuccessorInputLattices.size()) {
      assert(false && "region argument inference returned the wrong arity");
      return setAllToEntryStates(nonSuccessorInputLattices);
    }

    for (auto [lattice, type] :
         llvm::zip(nonSuccessorInputLattices, inferredTypes))
      if (type)
        join(lattice, TypeFact::concrete(type));
  }

protected:
  LogicalResult visitCallOperation(
      CallOpInterface call,
      ArrayRef<const dataflow::AbstractSparseLattice *> operandLattices,
      ArrayRef<dataflow::AbstractSparseLattice *> resultLattices) override {
    if (!isa<HCCallOp>(call.getOperation()))
      return AbstractSparseForwardDataFlowAnalysis::visitCallOperation(
          call, operandLattices, resultLattices);

    for (auto *rawLattice : resultLattices) {
      auto *lattice = static_cast<HCTypeLattice *>(rawLattice);
      join(lattice, factFromExistingType(lattice->getAnchor().getType()));
    }
    if (auto callee = dyn_cast_or_null<HCFuncOp>(call.resolveCallable())) {
      if (std::optional<FunctionType> fnType = callee.getFunctionType()) {
        for (auto [result, type] :
             llvm::zip(resultLattices, fnType->getResults()))
          if (!isHCUndefType(type))
            join(static_cast<HCTypeLattice *>(result),
                 TypeFact::concrete(type));
      }
    }

    // HC helpers are module-local in practice today, but they don't carry MLIR
    // private visibility yet. Use the visible return sites instead of letting a
    // public symbol's unknown external predecessors erase all useful facts.
    ProgramPoint *point = getProgramPointAfter(call);
    const auto *predecessors =
        getOrCreateFor<dataflow::PredecessorState>(point, point);
    for (Operation *predecessor : predecessors->getKnownPredecessors()) {
      for (auto [operand, result] :
           llvm::zip(predecessor->getOperands(), resultLattices)) {
        const auto *operandLattice = static_cast<const HCTypeLattice *>(
            AbstractSparseForwardDataFlowAnalysis::getLatticeElementFor(
                point, operand));
        join(static_cast<HCTypeLattice *>(result),
             factFromLattice(operandLattice));
      }
    }
    return success();
  }

  void visitCallableOperation(
      CallableOpInterface callable,
      ArrayRef<dataflow::AbstractSparseLattice *> argLattices) override {
    if (!isa<HCFuncOp>(callable.getOperation()))
      return AbstractSparseForwardDataFlowAnalysis::visitCallableOperation(
          callable, argLattices);

    for (auto *rawLattice : argLattices) {
      auto *lattice = static_cast<HCTypeLattice *>(rawLattice);
      join(lattice, factFromExistingType(lattice->getAnchor().getType()));
    }

    Region *region = callable.getCallableRegion();
    if (!region || region->empty())
      return AbstractSparseForwardDataFlowAnalysis::setAllToEntryStates(
          argLattices);

    ProgramPoint *point = getProgramPointBefore(&region->front());
    const auto *callsites = getOrCreateFor<dataflow::PredecessorState>(
        point, getProgramPointAfter(callable));
    // Same visibility story as call results: use known in-module callsites
    // even when public helper symbols make the generic framework conservative.
    for (Operation *callsite : callsites->getKnownPredecessors()) {
      auto call = cast<CallOpInterface>(callsite);
      for (auto [operand, arg] :
           llvm::zip(call.getArgOperands(), argLattices)) {
        const auto *operandLattice = static_cast<const HCTypeLattice *>(
            AbstractSparseForwardDataFlowAnalysis::getLatticeElementFor(
                point, operand));
        join(static_cast<HCTypeLattice *>(arg),
             factFromLattice(operandLattice));
      }
    }
  }

private:
  void join(HCTypeLattice *lattice, TypeFact fact) {
    propagateIfChanged(lattice, lattice->join(fact));
  }
};

static bool isNestedUnderHCCallable(Operation *op) {
  while (op) {
    if (isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(op))
      return true;
    op = op->getParentOp();
  }
  return false;
}

static Operation *nearestHCCallable(Operation *op) {
  while (op) {
    if (isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(op))
      return op;
    op = op->getParentOp();
  }
  return nullptr;
}

static bool isNestedUnderHCCallable(Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return isNestedUnderHCCallable(result.getOwner());
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isNestedUnderHCCallable(arg.getOwner()->getParentOp());
  return false;
}

static std::optional<Type> inferredTypeFor(DataFlowSolver &solver,
                                           Value value) {
  const HCTypeLattice *lattice = solver.lookupState<HCTypeLattice>(value);
  if (!lattice)
    return std::nullopt;
  const TypeFact &fact = lattice->getValue();
  if (!fact.hasUsableType())
    return std::nullopt;
  return fact.type;
}

static LogicalResult reportConflict(Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    return result.getOwner()->emitOpError()
           << "has conflicting HC type facts for result #"
           << result.getResultNumber();
  }

  auto arg = cast<BlockArgument>(value);
  Operation *owner = arg.getOwner()->getParentOp();
  if (owner)
    return owner->emitOpError()
           << "has conflicting HC type facts for block argument #"
           << arg.getArgNumber();
  return failure();
}

static LogicalResult reportTypeConflicts(ModuleOp module,
                                         DataFlowSolver &solver) {
  bool foundConflict = false;
  auto checkValue = [&](Value value) {
    if (!isNestedUnderHCCallable(value))
      return;
    const HCTypeLattice *lattice = solver.lookupState<HCTypeLattice>(value);
    if (!lattice || !lattice->getValue().isHardConflict())
      return;
    foundConflict = true;
    (void)reportConflict(value);
  };

  module.walk([&](Operation *op) {
    for (OpResult result : op->getResults())
      checkValue(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          checkValue(arg);
  });
  return failure(foundConflict);
}

template <typename CallableOpT>
static bool refineCallableResultType(CallableOpT op, unsigned index,
                                     Type inferred) {
  auto fnTypeAttr = op.getFunctionTypeAttr();
  if (!fnTypeAttr)
    return false;
  auto oldType = cast<FunctionType>(fnTypeAttr.getValue());
  if (index >= oldType.getNumResults() ||
      !shouldRefineHCType(oldType.getResult(index), inferred))
    return false;

  SmallVector<Type> results(oldType.getResults());
  results[index] = inferred;
  op.setFunctionTypeAttr(TypeAttr::get(
      FunctionType::get(op.getContext(), oldType.getInputs(), results)));
  return true;
}

static bool refineReturnedValueUsers(Value value, Type inferred) {
  bool changed = false;
  for (OpOperand &use : value.getUses()) {
    auto ret = dyn_cast<HCReturnOp>(use.getOwner());
    if (!ret)
      continue;
    Operation *callable = nearestHCCallable(ret.getOperation());
    if (auto kernel = dyn_cast_or_null<HCKernelOp>(callable))
      changed |=
          refineCallableResultType(kernel, use.getOperandNumber(), inferred);
    else if (auto func = dyn_cast_or_null<HCFuncOp>(callable))
      changed |=
          refineCallableResultType(func, use.getOperandNumber(), inferred);
    else if (auto intrinsic = dyn_cast_or_null<HCIntrinsicOp>(callable))
      changed |=
          refineCallableResultType(intrinsic, use.getOperandNumber(), inferred);
  }
  return changed;
}

static bool refineValue(Value value, Type inferred) {
  bool changed = refineReturnedValueUsers(value, inferred);
  if (!shouldRefineHCType(value.getType(), inferred))
    return changed;
  value.setType(inferred);
  return true;
}

static TypeFact factFromValue(DataFlowSolver &solver, Value value) {
  TypeFact fact = factFromExistingType(value.getType());
  if (std::optional<Type> inferred = inferredTypeFor(solver, value)) {
    TypeFact inferredFact = TypeFact::concrete(*inferred);
    if (!fact.hasUsableType() || shouldRefineHCType(fact.type, *inferred))
      return inferredFact;
    return joinExistingFacts(fact, inferredFact);
  }
  return fact;
}

static bool updateYieldedRegionResultTypes(ModuleOp module,
                                           DataFlowSolver &solver) {
  bool changed = false;
  module.walk([&](HCYieldedResultsOpInterface op) {
    ValueRange yieldedValues = op.getYieldedResultValues();
    if (yieldedValues.empty())
      return;
    if (yieldedValues.size() != op->getNumResults()) {
      assert(false && "verified yielded-result region arity mismatch");
      return;
    }
    for (auto [result, yielded] :
         llvm::zip_equal(op->getResults(), yieldedValues)) {
      TypeFact fact = factFromValue(solver, yielded);
      if (!fact.hasUsableType())
        continue;
      changed |= refineValue(result, fact.type);
    }
  });
  return changed;
}

template <typename CallableOpT>
static bool updateCallableFunctionType(CallableOpT op, DataFlowSolver &solver) {
  auto fnTypeAttr = op.getFunctionTypeAttr();
  if (!fnTypeAttr || op.getBody().empty())
    return false;

  auto oldType = cast<FunctionType>(fnTypeAttr.getValue());
  SmallVector<Type> inputs(op.getBody().front().getArgumentTypes());
  SmallVector<TypeFact> resultFacts;
  resultFacts.reserve(oldType.getNumResults());
  for (Type result : oldType.getResults())
    resultFacts.push_back(factFromExistingType(result));

  op.getBody().walk([&](HCReturnOp ret) {
    if (nearestHCCallable(ret.getOperation()) != op.getOperation())
      return;
    for (auto [idx, value] : llvm::enumerate(ret.getValues())) {
      if (idx >= resultFacts.size())
        return;
      TypeFact fact = factFromValue(solver, value);
      if (resultFacts[idx].hasUsableType() && fact.hasUsableType() &&
          shouldRefineHCType(resultFacts[idx].type, fact.type))
        resultFacts[idx] = fact;
      else
        resultFacts[idx] = joinExistingFacts(resultFacts[idx], fact);
    }
  });

  SmallVector<Type> results;
  results.reserve(resultFacts.size());
  for (auto [oldResult, fact] : llvm::zip(oldType.getResults(), resultFacts))
    results.push_back(fact.hasUsableType() ? fact.type : oldResult);

  auto newType = FunctionType::get(op.getContext(), inputs, results);
  if (newType == oldType)
    return false;
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  return true;
}

static bool updateCallableFunctionTypes(ModuleOp module,
                                        DataFlowSolver &solver) {
  bool changed = false;
  module.walk([&](Operation *op) {
    if (auto kernel = dyn_cast<HCKernelOp>(op))
      changed |= updateCallableFunctionType(kernel, solver);
    else if (auto func = dyn_cast<HCFuncOp>(op))
      changed |= updateCallableFunctionType(func, solver);
    else if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op))
      changed |= updateCallableFunctionType(intrinsic, solver);
  });
  return changed;
}

static bool applyInferredTypes(ModuleOp module, DataFlowSolver &solver) {
  bool changed = updateCallableFunctionTypes(module, solver);
  // Region result refinement can itself update enclosing callable signatures
  // through return users; the outer solver loop reruns until those surfaces
  // converge.
  changed |= updateYieldedRegionResultTypes(module, solver);
  module.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      if (!isNestedUnderHCCallable(result))
        continue;
      if (std::optional<Type> inferred = inferredTypeFor(solver, result))
        changed |= refineValue(result, *inferred);
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isNestedUnderHCCallable(arg))
            continue;
          if (std::optional<Type> inferred = inferredTypeFor(solver, arg))
            changed |= refineValue(arg, *inferred);
        }
      }
    }
  });
  return changed;
}

static void collectSyntheticJoinSymbol(StringRef name, llvm::StringSet<> &seen,
                                       SmallVectorImpl<std::string> &symbols) {
  if (isSyntheticJoinSymbolName(name) && seen.insert(name).second)
    symbols.push_back(name.str());
}

static void collectSyntheticJoinSymbols(Type type, llvm::StringSet<> &seen,
                                        SmallVectorImpl<std::string> &symbols) {
  if (!type)
    return;

  if (auto idx = dyn_cast<IdxType>(type)) {
    if (ExprAttr expr = idx.getExpr())
      sym::walkSymbolNames(expr.getValue(), [&](StringRef name) {
        collectSyntheticJoinSymbol(name, seen, symbols);
      });
    return;
  }
  if (auto pred = dyn_cast<PredType>(type)) {
    if (PredAttr predicate = pred.getPred())
      sym::walkSymbolNames(predicate.getValue(), [&](StringRef name) {
        collectSyntheticJoinSymbol(name, seen, symbols);
      });
    return;
  }
  if (auto tuple = dyn_cast<TupleType>(type)) {
    for (Type element : tuple.getTypes())
      collectSyntheticJoinSymbols(element, seen, symbols);
    return;
  }
  if (auto slice = dyn_cast<SliceType>(type)) {
    collectSyntheticJoinSymbols(slice.getLowerType(), seen, symbols);
    collectSyntheticJoinSymbols(slice.getUpperType(), seen, symbols);
    collectSyntheticJoinSymbols(slice.getStepType(), seen, symbols);
  }
}

template <typename CallableOpT>
static void collectSyntheticJoinSymbols(CallableOpT op, llvm::StringSet<> &seen,
                                        SmallVectorImpl<std::string> &symbols) {
  if (auto fnTypeAttr = op.getFunctionTypeAttr()) {
    auto fnType = cast<FunctionType>(fnTypeAttr.getValue());
    for (Type input : fnType.getInputs())
      collectSyntheticJoinSymbols(input, seen, symbols);
    for (Type result : fnType.getResults())
      collectSyntheticJoinSymbols(result, seen, symbols);
  }
}

using SymbolSubstitution = std::pair<std::string, std::string>;

static FailureOr<ixs_node *>
rewriteIxsNode(MLIRContext *ctx, const ixs_node *node,
               ArrayRef<SymbolSubstitution> substitutions) {
  auto *dialect = ctx->getOrLoadDialect<HCDialect>();
  sym::Store &store = dialect->getSymbolStore();
  SmallVector<ixs_node *> targets;
  SmallVector<ixs_node *> replacements;
  targets.reserve(substitutions.size());
  replacements.reserve(substitutions.size());
  for (const auto &[from, to] : substitutions) {
    FailureOr<sym::ExprHandle> target = sym::parseExpr(store, from);
    FailureOr<sym::ExprHandle> replacement = sym::parseExpr(store, to);
    if (failed(target) || failed(replacement))
      return failure();
    targets.push_back(const_cast<ixs_node *>(target->raw()));
    replacements.push_back(const_cast<ixs_node *>(replacement->raw()));
  }

  sym::Session session(store);
  ixs_node *rewritten =
      ixs_subs_multi(session.raw(), const_cast<ixs_node *>(node),
                     static_cast<uint32_t>(substitutions.size()),
                     targets.data(), replacements.data());
  if (!rewritten)
    return failure();
  return rewritten;
}

static FailureOr<ExprAttr>
rewriteExprAttr(MLIRContext *ctx, ExprAttr attr,
                ArrayRef<SymbolSubstitution> substitutions) {
  if (substitutions.empty())
    return attr;

  FailureOr<ixs_node *> rewritten =
      rewriteIxsNode(ctx, attr.getNode(), substitutions);
  if (failed(rewritten) || !ixs_node_is_expr(*rewritten))
    return failure();
  return ExprAttr::get(ctx, sym::ExprHandle(*rewritten));
}

static FailureOr<PredAttr>
rewritePredAttr(MLIRContext *ctx, PredAttr attr,
                ArrayRef<SymbolSubstitution> substitutions) {
  if (substitutions.empty())
    return attr;

  FailureOr<ixs_node *> rewritten =
      rewriteIxsNode(ctx, attr.getNode(), substitutions);
  if (failed(rewritten) || !ixs_node_is_pred(*rewritten))
    return failure();
  return PredAttr::get(ctx, sym::PredHandle(*rewritten));
}

static FailureOr<Type>
rewriteSyntheticJoinSymbols(Type type,
                            ArrayRef<SymbolSubstitution> substitutions) {
  if (!type)
    return type;

  MLIRContext *ctx = type.getContext();
  if (auto idx = dyn_cast<IdxType>(type)) {
    ExprAttr expr = idx.getExpr();
    if (!expr)
      return type;
    FailureOr<ExprAttr> rewritten = rewriteExprAttr(ctx, expr, substitutions);
    if (failed(rewritten))
      return failure();
    return IdxType::get(ctx, *rewritten);
  }
  if (auto pred = dyn_cast<PredType>(type)) {
    PredAttr predicate = pred.getPred();
    if (!predicate)
      return type;
    FailureOr<PredAttr> rewritten =
        rewritePredAttr(ctx, predicate, substitutions);
    if (failed(rewritten))
      return failure();
    return PredType::get(ctx, *rewritten);
  }
  if (auto tuple = dyn_cast<TupleType>(type)) {
    SmallVector<Type> elements;
    elements.reserve(tuple.size());
    bool changed = false;
    for (Type element : tuple.getTypes()) {
      FailureOr<Type> rewritten =
          rewriteSyntheticJoinSymbols(element, substitutions);
      if (failed(rewritten))
        return failure();
      changed |= *rewritten != element;
      elements.push_back(*rewritten);
    }
    if (!changed)
      return type;
    return TupleType::get(ctx, elements);
  }
  if (auto slice = dyn_cast<SliceType>(type)) {
    FailureOr<Type> lower =
        rewriteSyntheticJoinSymbols(slice.getLowerType(), substitutions);
    if (failed(lower))
      return failure();
    FailureOr<Type> upper =
        rewriteSyntheticJoinSymbols(slice.getUpperType(), substitutions);
    if (failed(upper))
      return failure();
    FailureOr<Type> step =
        rewriteSyntheticJoinSymbols(slice.getStepType(), substitutions);
    if (failed(step))
      return failure();
    if (*lower == slice.getLowerType() && *upper == slice.getUpperType() &&
        *step == slice.getStepType())
      return type;
    return SliceType::get(ctx, *lower, *upper, *step);
  }
  return type;
}

static FailureOr<FunctionType>
rewriteFunctionType(FunctionType oldType,
                    ArrayRef<SymbolSubstitution> substitutions) {
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  bool changed = false;
  inputs.reserve(oldType.getNumInputs());
  results.reserve(oldType.getNumResults());
  for (Type input : oldType.getInputs()) {
    FailureOr<Type> rewritten =
        rewriteSyntheticJoinSymbols(input, substitutions);
    if (failed(rewritten))
      return failure();
    changed |= *rewritten != input;
    inputs.push_back(*rewritten);
  }
  for (Type result : oldType.getResults()) {
    FailureOr<Type> rewritten =
        rewriteSyntheticJoinSymbols(result, substitutions);
    if (failed(rewritten))
      return failure();
    changed |= *rewritten != result;
    results.push_back(*rewritten);
  }
  if (!changed)
    return oldType;
  return FunctionType::get(oldType.getContext(), inputs, results);
}

template <typename CallableOpT>
static LogicalResult
rewriteCallableFunctionType(CallableOpT op,
                            ArrayRef<SymbolSubstitution> substitutions) {
  auto fnTypeAttr = op.getFunctionTypeAttr();
  if (!fnTypeAttr)
    return success();
  auto oldType = cast<FunctionType>(fnTypeAttr.getValue());
  FailureOr<FunctionType> newType = rewriteFunctionType(oldType, substitutions);
  if (failed(newType))
    return failure();
  if (*newType != oldType)
    op.setFunctionTypeAttr(TypeAttr::get(*newType));
  return success();
}

static LogicalResult
renumberSyntheticJoinSymbols(ModuleOp module,
                             ArrayRef<SymbolSubstitution> substitutions) {
  if (substitutions.empty())
    return success();

  WalkResult walkStatus = module.walk([&](Operation *op) -> WalkResult {
    if (auto kernel = dyn_cast<HCKernelOp>(op)) {
      if (failed(rewriteCallableFunctionType(kernel, substitutions)))
        return WalkResult::interrupt();
    } else if (auto func = dyn_cast<HCFuncOp>(op)) {
      if (failed(rewriteCallableFunctionType(func, substitutions)))
        return WalkResult::interrupt();
    } else if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op)) {
      if (failed(rewriteCallableFunctionType(intrinsic, substitutions)))
        return WalkResult::interrupt();
    }

    for (OpResult opResult : op->getResults()) {
      FailureOr<Type> newType =
          rewriteSyntheticJoinSymbols(opResult.getType(), substitutions);
      if (failed(newType))
        return WalkResult::interrupt();
      opResult.setType(*newType);
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          FailureOr<Type> newType =
              rewriteSyntheticJoinSymbols(arg.getType(), substitutions);
          if (failed(newType))
            return WalkResult::interrupt();
          arg.setType(*newType);
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(walkStatus.wasInterrupted());
}

static LogicalResult renumberSyntheticJoinSymbols(ModuleOp module) {
  llvm::StringSet<> seen;
  SmallVector<std::string> symbols;
  module.walk([&](Operation *op) {
    if (auto kernel = dyn_cast<HCKernelOp>(op))
      collectSyntheticJoinSymbols(kernel, seen, symbols);
    else if (auto func = dyn_cast<HCFuncOp>(op))
      collectSyntheticJoinSymbols(func, seen, symbols);
    else if (auto intrinsic = dyn_cast<HCIntrinsicOp>(op))
      collectSyntheticJoinSymbols(intrinsic, seen, symbols);

    for (OpResult result : op->getResults())
      collectSyntheticJoinSymbols(result.getType(), seen, symbols);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          collectSyntheticJoinSymbols(arg.getType(), seen, symbols);
  });

  SmallVector<SymbolSubstitution> substitutions;
  substitutions.reserve(symbols.size());
  for (auto [index, symbol] : llvm::enumerate(symbols)) {
    SmallString<16> stable(kSyntheticJoinPrefix);
    stable += Twine(index).str();
    substitutions.emplace_back(symbol, stable.str().str());
  }
  return renumberSyntheticJoinSymbols(module, substitutions);
}

struct HCInferTypesPass : public hc::impl::HCInferTypesBase<HCInferTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    while (true) {
      DataFlowSolver solver;
      dataflow::loadBaselineAnalyses(solver);
      solver.load<HCTypeAnalysis>();
      if (failed(solver.initializeAndRun(module)))
        return signalPassFailure();
      if (failed(reportTypeConflicts(module, solver)))
        return signalPassFailure();
      if (!applyInferredTypes(module, solver)) {
        if (failed(renumberSyntheticJoinSymbols(module)))
          return signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

// `createHCInferTypesPass()` is emitted by tablegen (friend of the
// `impl::HCInferTypesBase` CRTP). See `Passes.td` — no `let constructor`, so
// the generated factory is the only one.
