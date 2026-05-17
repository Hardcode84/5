// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-specialize-literals`. Folds `literal_bindings`
// (stamped by `hc.compile(symbols={K: 4, ...})` on `hc.kernel`) into
// IR: substitute each named symbol with its integer value in every
// reachable `#hc.expr` / `#hc.pred`, rebuild surrounding shape /
// layout / shaped-type stack via `AttrTypeReplacer`.
//
// Bindings live in IR (not a pass option) so a `hc_ir_text` snapshot
// is self-contained -- re-running on the snapshot reproduces the
// specialized IR. `literals` declares what may be bound;
// `literal_bindings` carries the bound values.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCSPECIALIZELITERALS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Cross-check binding keys against the `literals` whitelist when present.
// Undeclared key = hard error. Absent `literals` is permissive.
static LogicalResult validateBindings(HCKernelOp kernel,
                                      DictionaryAttr bindings) {
  ArrayAttr literals = kernel.getLiteralsAttr();
  if (!literals)
    return success();
  llvm::StringSet<llvm::MallocAllocator> declared;
  for (StringAttr name : literals.getAsRange<StringAttr>())
    declared.insert(name.getValue());
  for (NamedAttribute entry : bindings) {
    StringRef name = entry.getName().getValue();
    if (declared.count(name))
      continue;
    // `$`-prefixed names are launch-context (seeded by the front-to-hc
    // handshake), not user literals -- skip the whitelist check.
    if (name.starts_with("$"))
      continue;
    return kernel->emitOpError("literal_bindings key '")
           << name << "' is not declared in `literals`";
  }
  return success();
}

// Build `ixs_subs_multi` inputs from `name -> IntegerAttr`. Fail-fast on
// the first invalid entry -- specialization is all-or-nothing per kernel.
static LogicalResult
composeSubstitutionPairs(HCKernelOp kernel, DictionaryAttr bindings,
                         sym::Store &store,
                         SmallVectorImpl<ixs_node *> &targets,
                         SmallVectorImpl<ixs_node *> &replacements) {
  targets.reserve(bindings.size());
  replacements.reserve(bindings.size());
  for (NamedAttribute entry : bindings) {
    auto valueAttr = dyn_cast<IntegerAttr>(entry.getValue());
    if (!valueAttr || !valueAttr.getType().isInteger())
      return kernel->emitOpError("literal_bindings['")
             << entry.getName().getValue() << "'] must be an IntegerAttr (got "
             << entry.getValue() << ")";
    auto symHandle = sym::composeExprSym(store, entry.getName().getValue());
    auto intHandle = sym::composeExprInt(store, valueAttr.getInt());
    if (failed(symHandle) || failed(intHandle))
      return kernel->emitOpError(
                 "failed to build symbolic substitution pair for '")
             << entry.getName().getValue() << "'";
    targets.push_back(const_cast<ixs_node *>(symHandle->raw()));
    replacements.push_back(const_cast<ixs_node *>(intHandle->raw()));
  }
  return success();
}

// Shared between `ExprHandle` / `PredHandle` -- identical substitution
// semantics.
template <typename Handle>
static Handle substituteHandle(sym::Session &session, Handle handle,
                               ArrayRef<ixs_node *> targets,
                               ArrayRef<ixs_node *> replacements) {
  if (!handle)
    return handle;
  ixs_node *result =
      ixs_subs_multi(session.raw(), const_cast<ixs_node *>(handle.raw()),
                     static_cast<uint32_t>(targets.size()), targets.data(),
                     replacements.data());
  return Handle(result ? result : handle.raw());
}

// `AttrTypeReplacer::recursivelyReplaceElementsIn` doesn't visit block
// argument types -- walk regions explicitly after the op tree.
static void retypeBlockArguments(Operation *op, AttrTypeReplacer &replacer) {
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (BlockArgument arg : block.getArguments()) {
        Type rewritten = replacer.replace(arg.getType());
        if (rewritten && rewritten != arg.getType())
          arg.setType(rewritten);
      }
}

// Substitute every `#hc.expr` / `#hc.pred` and through them every
// shape / layout / shaped type. `replaceTypes=true` covers block-arg types.
static void specializeKernel(HCKernelOp kernel, sym::Store &store,
                             ArrayRef<ixs_node *> targets,
                             ArrayRef<ixs_node *> replacements) {
  sym::Session session(store);

  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](ExprAttr attr) -> std::pair<Attribute, WalkResult> {
        sym::ExprHandle replaced =
            substituteHandle(session, attr.getValue(), targets, replacements);
        if (replaced == attr.getValue())
          return {attr, WalkResult::skip()};
        // `ExprAttr` is a leaf -- opaque `ixs_node *`, not an MLIR sub-element.
        return {ExprAttr::get(attr.getContext(), replaced), WalkResult::skip()};
      });
  replacer.addReplacement(
      [&](PredAttr attr) -> std::pair<Attribute, WalkResult> {
        sym::PredHandle replaced =
            substituteHandle(session, attr.getValue(), targets, replacements);
        if (replaced == attr.getValue())
          return {attr, WalkResult::skip()};
        return {PredAttr::get(attr.getContext(), replaced), WalkResult::skip()};
      });

  // Walk the kernel op so `function_type` stays parity with block args
  // (verifier enforces). `literal_bindings` holds only `IntegerAttr` --
  // no Expr/Pred matches, safe to recurse through.
  replacer.recursivelyReplaceElementsIn(kernel, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
  retypeBlockArguments(kernel, replacer);

  // Drop consumed bindings (re-run = no-op). `literals` stays -- declaration.
  kernel.removeLiteralBindingsAttr();
}

struct HCSpecializeLiteralsPass
    : public hc::impl::HCSpecializeLiteralsBase<HCSpecializeLiteralsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

    WalkResult status =
        getOperation()->walk([&](HCKernelOp kernel) -> WalkResult {
          DictionaryAttr bindings = kernel.getLiteralBindingsAttr();
          if (!bindings)
            return WalkResult::advance();
          // Empty bindings -- drop attribute, skip substitution.
          if (bindings.empty()) {
            kernel.removeLiteralBindingsAttr();
            return WalkResult::advance();
          }
          if (failed(validateBindings(kernel, bindings)))
            return WalkResult::interrupt();
          SmallVector<ixs_node *, 4> targets;
          SmallVector<ixs_node *, 4> replacements;
          if (failed(composeSubstitutionPairs(kernel, bindings, store, targets,
                                              replacements)))
            return WalkResult::interrupt();
          specializeKernel(kernel, store, targets, replacements);
          return WalkResult::advance();
        });
    if (status.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
