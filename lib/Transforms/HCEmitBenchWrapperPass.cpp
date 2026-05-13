// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-emit-bench-wrapper`. Runs after
// `-hc-lower-launch-func-to-runtime` has produced the host wrapper as
// an `llvm.func` containing a single `hc_rt_launch_kernel` call plus
// the per-callsite `_data` / `_handle` / kernel-name globals. For each
// such wrapper, we clone the whole function into a sibling
// `<name>_bench`, append an `i64 %n_inner` block arg, swap the
// runtime call to `hc_rt_launch_kernel_repeat` with `%n_inner` as the
// trailing argument, and rewrite the terminator to return the i64
// elapsed-ns the repeat entry produces.
//
// Cloning the post-runtime form (rather than running before
// `-hc-lower-launch-func-to-runtime` on the `gpu.launch_func`) gives us
// shared cache slots for free: cloning preserves the `llvm.addressof`
// SymbolRefAttrs verbatim, so both wrappers' `hc_rt_load_kernel` calls
// hand the same `<kernel>_handle` pointer to the runtime — a warm-up
// invocation on either side primes the cache for the other.

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

// Locate the unique `hc_rt_launch_kernel` callsite inside `func`.
// Returns null if zero or more-than-one matches are found — both are
// shapes we don't expect for a host wrapper, and either way the clone
// would be ambiguous, so we skip cleanly rather than guessing.
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

// Materialize the `hc_rt_launch_kernel_repeat` decl at module scope if
// it isn't there already. The repeat signature mirrors the
// `hc_rt_launch_kernel` decl that `hc-lower-launch-func-to-runtime`
// planted earlier — same arg types, plus a trailing `i64`, with an
// `i64` return instead of void — so we look up the existing launch
// decl and append a slot rather than respelling the ABI from scratch.
// Keeps the two decls structurally locked: any future addition to the
// launch signature surfaces here as an automatic widening of the
// repeat signature.
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

// Clone `host` into a `<name>_bench` sibling and rewire the launch call
// + return. The clone preserves global symbol references verbatim, so
// the per-callsite cache slot is shared between the two wrappers.
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

  // Update the cloned function's signature: append i64, return i64.
  auto oldFnType = benchFunc.getFunctionType();
  SmallVector<Type> newParams(oldFnType.getParams());
  newParams.push_back(i64Type);
  auto newFnType = LLVM::LLVMFunctionType::get(i64Type, newParams);
  benchFunc.setFunctionType(newFnType);

  // Add the matching block argument to the entry block.
  Block &entry = benchFunc.getBody().front();
  Value nInner = entry.addArgument(i64Type, host.getLoc());

  LLVM::CallOp clonedLaunch = findLaunchCall(benchFunc);
  assert(clonedLaunch && "clone lost the launch call");

  LLVM::LLVMFuncOp repeatDecl = getOrInsertRepeatDecl(mod, builder);

  // Build the repeat call: same args + n_inner. The kernelParams
  // alloca/insertvalue chain was cloned along with the function body,
  // so the `void**` we hand to repeat is the same `void**` the
  // single-shot launch would use.
  builder.setInsertionPoint(clonedLaunch);
  SmallVector<Value> repeatArgs(clonedLaunch.getArgOperands());
  repeatArgs.push_back(nInner);
  auto repeatCall = LLVM::CallOp::create(builder, clonedLaunch.getLoc(),
                                         repeatDecl, repeatArgs);
  clonedLaunch.erase();

  // Replace the cloned `llvm.return` (void) with `llvm.return %t : i64`.
  // The original wrapper has exactly one terminator (an llvm.return
  // with no operands); the clone inherits that shape.
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

  // Snapshot the candidate wrappers before mutating the module — we
  // append the bench clone as a sibling of `host`, so walking and
  // mutating concurrently would double-count freshly-minted clones.
  // Skips:
  //   * external declarations — runtime decls aren't bodies we can clone.
  //   * `<name>_bench` symbols — keep us from re-bench'ing our own
  //     output.
  //   * wrappers whose `<name>_bench` sibling already exists — a prior
  //     run already minted it, this run is a no-op for that wrapper.
  //   * wrappers without an `hc_rt_launch_kernel` call — helper
  //     unpackers and other non-launch llvm.funcs.
  // Together those rules make the pass idempotent and tolerant of
  // partially-bench'd modules (mixing fresh and already-processed
  // wrappers in the same module is fine).
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
