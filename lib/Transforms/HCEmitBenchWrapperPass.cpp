// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-emit-bench-wrapper`. Clones each host wrapper that
// calls `hc_rt_launch_kernel` into a `<name>_bench` sibling with a
// trailing `i64 %n_inner` block arg, swaps the call for
// `hc_rt_launch_kernel_repeat` (taking `%n_inner`), and returns the
// i64 elapsed-ns the repeat entry produces.
//
// Cloning preserves `llvm.addressof` SymbolRefAttrs verbatim, so both
// wrappers share `<kernel>_handle` cache slots — warm-up on either
// side primes the other.

#include "hc/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCEMITBENCHWRAPPER
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kLaunchSymbol = "hc_rt_launch_kernel";
static constexpr llvm::StringLiteral kLaunchRepeatSymbol =
    "hc_rt_launch_kernel_repeat";
static constexpr llvm::StringLiteral kBenchSuffix = "_bench";

// Null if zero or >1 callsites — ambiguous shape, skip cleanly.
static LLVM::CallOp findLaunchCall(LLVM::LLVMFuncOp func) {
  LLVM::CallOp found;
  bool ambiguous = false;
  func.walk([&](LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee || *callee != kLaunchSymbol)
      return;
    if (found) {
      ambiguous = true;
      return;
    }
    found = call;
  });
  return ambiguous ? nullptr : found;
}

// Derive repeat signature from the existing `hc_rt_launch_kernel` decl
// (+ trailing i64, i64 return) so the two decls stay structurally locked.
static LLVM::LLVMFuncOp getOrInsertRepeatDecl(ModuleOp mod,
                                              OpBuilder &builder) {
  SymbolTable table(mod);
  if (auto existing = table.lookup<LLVM::LLVMFuncOp>(kLaunchRepeatSymbol))
    return existing;

  auto launchDecl = table.lookup<LLVM::LLVMFuncOp>(kLaunchSymbol);
  assert(
      launchDecl &&
      "expected hc_rt_launch_kernel decl from hc-lower-launch-func-to-runtime "
      "before hc-emit-bench-wrapper runs");

  MLIRContext *ctx = mod.getContext();
  Type i64Type = IntegerType::get(ctx, 64);
  SmallVector<Type> params(launchDecl.getFunctionType().getParams());
  params.push_back(i64Type);
  auto repeatType = LLVM::LLVMFunctionType::get(i64Type, params);

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(mod.getBody());
  return LLVM::LLVMFuncOp::create(builder, mod.getLoc(), kLaunchRepeatSymbol,
                                  repeatType);
}

// Clone preserves SymbolRefAttrs verbatim — wrappers share cache slots.
static void emitBenchClone(LLVM::LLVMFuncOp host, ModuleOp mod) {
  MLIRContext *ctx = host.getContext();
  Type i64Type = IntegerType::get(ctx, 64);

  LLVM::CallOp launchCall = findLaunchCall(host);
  if (!launchCall)
    return;

  OpBuilder builder(host);
  builder.setInsertionPointAfter(host);
  auto benchFunc = cast<LLVM::LLVMFuncOp>(builder.clone(*host.getOperation()));
  benchFunc.setSymName((host.getSymName() + kBenchSuffix).str());

  auto oldFnType = benchFunc.getFunctionType();
  SmallVector<Type> newParams(oldFnType.getParams());
  newParams.push_back(i64Type);
  auto newFnType = LLVM::LLVMFunctionType::get(i64Type, newParams);
  benchFunc.setFunctionType(newFnType);

  Block &entry = benchFunc.getBody().front();
  Value nInner = entry.addArgument(i64Type, host.getLoc());

  LLVM::CallOp clonedLaunch = findLaunchCall(benchFunc);
  assert(clonedLaunch && "clone lost the launch call");

  LLVM::LLVMFuncOp repeatDecl = getOrInsertRepeatDecl(mod, builder);

  // kernelParams alloca/insertvalue chain cloned with body — same `void**`.
  builder.setInsertionPoint(clonedLaunch);
  SmallVector<Value> repeatArgs(clonedLaunch.getArgOperands());
  repeatArgs.push_back(nInner);
  auto repeatCall = LLVM::CallOp::create(builder, clonedLaunch.getLoc(),
                                         repeatDecl, repeatArgs);
  clonedLaunch.erase();

  // Original wrapper has exactly one `llvm.return` terminator.
  LLVM::ReturnOp oldRet;
  benchFunc.walk([&](LLVM::ReturnOp r) { oldRet = r; });
  assert(oldRet && "cloned bench wrapper has no llvm.return terminator");

  builder.setInsertionPoint(oldRet);
  LLVM::ReturnOp::create(builder, oldRet.getLoc(), repeatCall.getResult());
  oldRet.erase();
}

class HCEmitBenchWrapperPass
    : public hc::impl::HCEmitBenchWrapperBase<HCEmitBenchWrapperPass> {
public:
  using Base::Base;
  void runOnOperation() final;
};

void HCEmitBenchWrapperPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // Snapshot before mutating; clones are siblings, walk-and-mutate would
  // double-count. Skip externals, `_bench` symbols, already-bench'd
  // wrappers, and any wrapper without an `hc_rt_launch_kernel` call.
  // Rules make the pass idempotent on partially-bench'd modules.
  SymbolTable table(mod);
  SmallVector<LLVM::LLVMFuncOp> hosts;
  mod.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal())
      return;
    StringRef name = func.getSymName();
    if (name.ends_with(kBenchSuffix))
      return;
    if (table.lookup<LLVM::LLVMFuncOp>((name + kBenchSuffix).str()))
      return;
    if (!findLaunchCall(func))
      return;
    hosts.push_back(func);
  });

  for (LLVM::LLVMFuncOp host : hosts)
    emitBenchClone(host, mod);
}

} // namespace
