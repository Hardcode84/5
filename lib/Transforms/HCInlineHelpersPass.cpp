// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-inline-helpers`, the HC-to-HC pass that consumes helper calls
// before scope-region normalization.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCINLINEHELPERS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static HCFuncOp lookupCallee(HCCallOp call) {
  return SymbolTable::lookupNearestSymbolFrom<HCFuncOp>(call.getOperation(),
                                                        call.getCalleeAttr());
}

static bool callIsRecursive(HCCallOp call, HCFuncOp callee) {
  Operation *parent = call->getParentOp();
  while (parent) {
    if (parent == callee.getOperation())
      return true;
    if (isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(parent))
      return false;
    parent = parent->getParentOp();
  }
  return false;
}

static LogicalResult inlineCall(HCCallOp call, OpBuilder &builder,
                                llvm::StringSet<> &inlinedCallees) {
  HCFuncOp callee = lookupCallee(call);
  if (!callee)
    return call.emitOpError("cannot inline unresolved hc.func callee");
  if (callIsRecursive(call, callee))
    return call.emitOpError("recursive hc.func inlining is not supported");
  if (callee.getBody().empty() || callee.getBody().front().empty())
    return call.emitOpError("cannot inline empty hc.func callee");

  Block &body = callee.getBody().front();
  auto ret = cast<HCReturnOp>(body.back());
  if (ret.getValues().size() != call.getNumResults())
    return call.emitOpError("callee return arity ")
           << ret.getValues().size() << " does not match call result arity "
           << call.getNumResults();
  if (body.getNumArguments() != call.getArgs().size())
    return call.emitOpError("callee argument arity ")
           << body.getNumArguments() << " does not match call argument arity "
           << call.getArgs().size();

  IRMapping mapping;
  for (auto [arg, value] : llvm::zip_equal(body.getArguments(), call.getArgs()))
    mapping.map(arg, value);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(call);
  for (Operation &op : body.without_terminator())
    builder.clone(op, mapping);

  SmallVector<Value> replacements;
  replacements.reserve(ret.getValues().size());
  for (Value returned : ret.getValues())
    replacements.push_back(mapping.lookupOrDefault(returned));
  call.replaceAllUsesWith(replacements);
  inlinedCallees.insert(callee.getName());
  call.erase();
  return success();
}

static LogicalResult inlineHelperCalls(Operation *root,
                                       llvm::StringSet<> &inlinedCallees) {
  OpBuilder builder(root->getContext());
  while (true) {
    HCCallOp nextCall;
    root->walk([&](HCCallOp call) {
      nextCall = call;
      return WalkResult::interrupt();
    });
    if (!nextCall)
      return success();
    if (failed(inlineCall(nextCall, builder, inlinedCallees)))
      return failure();
  }
}

static void eraseUnreferencedInlinedHelpers(Operation *root,
                                            llvm::StringSet<> &inlinedCallees) {
  if (inlinedCallees.empty())
    return;

  llvm::StringSet<> referenced;
  root->walk([&](HCCallOp call) { referenced.insert(call.getCallee()); });

  SmallVector<HCFuncOp> funcs;
  root->walk([&](HCFuncOp func) {
    if (inlinedCallees.contains(func.getName()) &&
        !referenced.contains(func.getName()))
      funcs.push_back(func);
  });
  for (HCFuncOp func : funcs)
    func.erase();
}

struct HCInlineHelpersPass
    : public hc::impl::HCInlineHelpersBase<HCInlineHelpersPass> {
  using Base::Base;

  void runOnOperation() override {
    llvm::StringSet<> inlinedCallees;
    if (failed(inlineHelperCalls(getOperation(), inlinedCallees)))
      return signalPassFailure();
    eraseUnreferencedInlinedHelpers(getOperation(), inlinedCallees);
  }
};

} // namespace

// `createHCInlineHelpersPass()` is emitted by tablegen.
