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
#include "hc/IR/HCTypes.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCINFERTYPES
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

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

  bool isUnknown() const { return kind == Kind::Unknown; }
  bool isConcrete() const { return kind == Kind::Concrete && type; }
  bool isConflict() const { return kind == Kind::Conflict; }

  bool operator==(const TypeFact &rhs) const {
    return kind == rhs.kind && type == rhs.type;
  }

  static TypeFact join(const TypeFact &lhs, const TypeFact &rhs) {
    if (lhs.isUnknown())
      return rhs;
    if (rhs.isUnknown())
      return lhs;
    if (lhs.isConflict() || rhs.isConflict())
      return conflict();
    if (lhs.type == rhs.type)
      return lhs;
    if (auto joinable = dyn_cast<HCJoinableTypeInterface>(lhs.type))
      if (Type common = joinable.joinHCType(rhs.type))
        return concrete(common);
    if (auto joinable = dyn_cast<HCJoinableTypeInterface>(rhs.type))
      if (Type common = joinable.joinHCType(lhs.type))
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
      os << "<conflict>";
      return;
    }
    os << type;
  }
};

struct HCTypeLattice : public dataflow::Lattice<TypeFact> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCTypeLattice)
  using Lattice::Lattice;
};

static bool isUndef(Type type) { return isa<UndefType>(type); }

static TypeFact factFromExistingType(Type type) {
  if (!type || isUndef(type))
    return TypeFact::unknown();
  return TypeFact::concrete(type);
}

static TypeFact factFromLattice(const HCTypeLattice *lattice) {
  if (!lattice)
    return TypeFact::unknown();
  return lattice->getValue();
}

static Type unpinnedIdx(MLIRContext *ctx) {
  return IdxType::get(ctx, ExprAttr{});
}

class HCTypeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<HCTypeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(HCTypeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(factFromExistingType(
                                    lattice->getAnchor().getType())));
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
        operandTypes.push_back(fact.isConcrete() ? fact.type : Type{});
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
    if (!isa<RegionBranchOpInterface>(op) || successor.isParent())
      return setAllToEntryStates(nonSuccessorInputLattices);

    // HC region-branch ops currently expose only loop induction variables as
    // non-forwarded region arguments. Preserve concrete annotations when they
    // exist, and refine pre-inference placeholders to the unpinned index type.
    for (auto [value, lattice] :
         llvm::zip(nonSuccessorInputs, nonSuccessorInputLattices)) {
      join(lattice, factFromExistingType(value.getType()));
      if (isUndef(value.getType()))
        join(lattice, TypeFact::concrete(unpinnedIdx(op->getContext())));
    }
  }

private:
  void join(HCTypeLattice *lattice, TypeFact fact) {
    propagateIfChanged(lattice, lattice->join(fact));
  }
};

static bool shouldRefine(Type current, Type inferred) {
  if (!inferred || current == inferred)
    return false;
  if (isUndef(current))
    return true;
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

static bool isNestedUnderHCCallable(Operation *op) {
  while (op) {
    if (isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(op))
      return true;
    op = op->getParentOp();
  }
  return false;
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
  if (!fact.isConcrete())
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
    if (!lattice || !lattice->getValue().isConflict())
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

static bool refineValue(Value value, Type inferred) {
  if (!shouldRefine(value.getType(), inferred))
    return false;
  value.setType(inferred);
  return true;
}

static bool applyInferredTypes(ModuleOp module, DataFlowSolver &solver) {
  bool changed = false;
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
      if (!applyInferredTypes(module, solver))
        return;
    }
  }
};

} // namespace

// `createHCInferTypesPass()` is emitted by tablegen (friend of the
// `impl::HCInferTypesBase` CRTP). See `Passes.td` — no `let constructor`, so
// the generated factory is the only one.
