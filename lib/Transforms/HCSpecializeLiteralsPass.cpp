// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-specialize-literals`. Folds the compile-time literal
// bindings stamped by the launcher (`hc.compile(symbols={K: 4, ...})` on the
// enclosing `hc.kernel`'s `literal_bindings` dict attr) into the IR by
// substituting each named symbol with its integer value in every reachable
// `#hc.expr` / `#hc.pred` payload and rebuilding the surrounding shape /
// layout / shaped-type stack via `AttrTypeReplacer`.
//
// Bindings live in the IR rather than on a pass option: the front IR
// snapshot inspected via `compile().hc_ir_text` is then self-contained
// (re-running the pass on the snapshot reproduces the specialized IR
// without any out-of-band state), `hc-opt -hc-specialize-literals` on a
// hand-written module just reads what's already there, and the
// declaration / valuation split lives where it belongs (`literals` says
// which symbols *can* be bound; `literal_bindings` says what they are
// bound to here).
//
// After the pass, dims, offsets, and storage_size expressions that were
// symbolic in K become integer-literal-ground, and every shape-sensitive
// downstream pass — `hc-verify-static-shapes`, `hc-decompose-shaped-values`,
// `hc-flatten-with-layouts`, `hc-lower-launch-body` — sees concrete dims
// without any new substitution plumbing inside each consumer.

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

// Cross-check every binding key against the kernel's declared `literals`
// whitelist when one is present. A binding for an undeclared symbol is a
// hard error here so the pipeline fails loud at the specialization point
// instead of silently folding a name the kernel never opted in to —
// matches the `hc.compile(symbols={...})` Python-side rejection of
// undeclared keys but covers IR-side stamping paths (LIT, hand-written
// modules) too. An absent `literals` list (legacy / IR-only test
// payloads) is permissive: nothing has been promised one way or the
// other, so any binding is accepted.
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
    // Launch-context symbols (`$WGS<axis>`, `$WS<axis>`, `$WV0`,
    // `$GSZ0`, `$STRIDE_<axis>_<arg>`, ...) are seeded by the front-
    // to-hc handshake from `group_shape` / `work_shape` /
    // `subgroup_size`, not by `hc.compile(symbols={...})`; the
    // `literals` whitelist is the user-facing specialization
    // contract and doesn't speak for system-managed names. Skip the
    // cross-check for them so a kernel that declared a user literal
    // (e.g. `literals = ["TILE"]`) still legally carries
    // launch-context bindings.
    if (name.starts_with("$"))
      continue;
    return kernel->emitOpError("literal_bindings key '")
           << name << "' is not declared in `literals`";
  }
  return success();
}

// Convert a `DictionaryAttr` of `name -> IntegerAttr` into a parallel pair
// of `ixs_node *` arrays (target symbol nodes, replacement int nodes) that
// `ixs_subs_multi` can consume. Reports the first offending entry on
// failure instead of accumulating diagnostics; specialization is all-or-
// nothing per kernel.
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

// Apply a list of (target, replacement) substitutions to a single symbolic
// handle. Templated so the same machinery serves both `ExprHandle` and
// `PredHandle` — the two have identical substitution semantics and
// re-implementing the body for each just invites them to drift apart.
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

// Block-argument types ride on each block (not on the parent op's
// attributes), so `AttrTypeReplacer::recursivelyReplaceElementsIn` does
// not visit them. Walk the regions explicitly after the op tree walk.
static void retypeBlockArguments(Operation *op, AttrTypeReplacer &replacer) {
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (BlockArgument arg : block.getArguments()) {
        Type rewritten = replacer.replace(arg.getType());
        if (rewritten && rewritten != arg.getType())
          arg.setType(rewritten);
      }
}

// Substitute every reachable `#hc.expr` / `#hc.pred` (and through them
// every `#hc.shape`, `#hc.layout`, `!hc.idx`, `!hc.pred`, shaped type)
// inside the kernel body. `replaceTypes=true` covers block-argument types
// — kernel-arg buffer shapes, region IV types — so no symbolic carrier
// for a bound name survives.
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
        // `ExprAttr` wraps an opaque `ixs_node *` (not an MLIR
        // sub-element), so the replacer would not recurse into it
        // even with `advance()`. `skip` documents that this rebuild
        // is the leaf — the parent (`ShapeAttr`, `LayoutAttr`,
        // `IdxType`, ...) takes over from here.
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

  // Walk the kernel op itself so its own attribute dictionary
  // (`function_type` in particular — block args and the declared
  // function type must stay in sync, the verifier checks parity) gets
  // rewritten alongside the body. The `literal_bindings` dict only
  // holds `IntegerAttr` values, so the `ExprAttr` / `PredAttr`
  // replacements registered above don't match anything inside it —
  // safe to recurse through.
  replacer.recursivelyReplaceElementsIn(kernel, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
  retypeBlockArguments(kernel, replacer);

  // Drop the consumed `literal_bindings` so a downstream re-run is a
  // no-op and a `hc_ir_text` snapshot doesn't suggest there is more
  // specialization to do. `literals` stays put — it's the declaration,
  // not the bound state.
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
          // Empty `literal_bindings = {}` is the trivial case — still drop
          // the attribute so a subsequent run / `hc_ir_text` snapshot is
          // clean, but skip the substitution machinery.
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
