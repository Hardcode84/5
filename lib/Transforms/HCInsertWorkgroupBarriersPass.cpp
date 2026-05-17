// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-insert-workgroup-barriers`: dedicated barrier emitter
// for the post-1st-launch-body pipeline slot. See the pass description
// in `include/hc/Transforms/Passes.td` and the contract note in
// `doc/lowering.md` "Workgroup-AS synchronization".

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCINSERTWORKGROUPBARRIERS
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// Workgroup-AS storage root reached by walking back from `v` through
// the operand chains an `hc.generic` boundary may carry. Returns the
// canonical SSA Value identifying the storage (typically the result of
// an `hc.alloc workgroup`) or {} when `v` doesn't reference workgroup-AS
// storage in the first place.
//
// Two operands resolve to the same root iff they share the same backing
// allocation — pointer equality on the returned Value is the comparison
// the per-block pending set keys on. We walk through:
//   * single-input `unrealized_conversion_cast` (the
//     `ptr<workgroup>` ↔ `bare_tensor` UCC sandwich the launch-body
//     type converter plants on collective candidates and which the
//     surrounding canonicalize/cse pair has not necessarily folded yet
//     at the schedule slot this pass lives at), and
//   * `hc.ptr_offset` (different offsets into the same allocation
//     trivially share storage; the alloc is the root either way).
//
// `bare_tensor` carriers reached during the walk are accepted because
// the launch-body converter only plants the UCC sandwich on
// workgroup-staged tiles — the same convention `lowerCollective` keys
// on. A `bare_tensor` outside a `gpu.launch` would not be a workgroup
// carrier, but this pass only runs on `gpu.launch` bodies, so the
// distinction collapses to "trace back through whatever the converter
// planted until we can't trace any further".
static Value resolveWorkgroupRoot(Value v) {
  bool workgroupCarrier = false;
  if (auto p = dyn_cast<PtrType>(v.getType()))
    workgroupCarrier = (p.getAddrSpace() == AddrSpace::Workgroup);
  else if (isa<BareTensorType>(v.getType()))
    workgroupCarrier = true;
  if (!workgroupCarrier)
    return {};

  Value current = v;
  while (Operation *def = current.getDefiningOp()) {
    if (auto ucc = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (ucc.getInputs().size() != 1)
        break;
      current = ucc.getInputs()[0];
      continue;
    }
    if (auto off = dyn_cast<HCPtrOffsetOp>(def)) {
      current = off.getSource();
      continue;
    }
    break;
  }
  return current;
}

// Read / write workgroup-AS storage roots an `hc.generic` touches.
// `ins` -> reads, `outs` -> writes. `outs` on `hc.generic` follows the
// "outs-as-init" contract, so the op also *reads* the initial value of
// each out — for the barrier decision that read happens at the same
// instant as the write, so collapsing both edges into "this op
// references this root" would not change the conservative answer here.
// Tracking writes separately is what lets a future precise pass elide
// barriers between two read-only generics on the same root without
// re-deriving the role.
struct WorkgroupRoots {
  llvm::SmallSetVector<Value, 4> reads;
  llvm::SmallSetVector<Value, 4> writes;
};

static WorkgroupRoots collectWorkgroupRoots(HCGenericOp op) {
  WorkgroupRoots roots;
  for (Value in : op.getIns())
    if (Value root = resolveWorkgroupRoot(in))
      roots.reads.insert(root);
  for (Value out : op.getOuts())
    if (Value root = resolveWorkgroupRoot(out))
      roots.writes.insert(root);
  return roots;
}

static bool intersects(const llvm::SmallSetVector<Value, 4> &a,
                       const llvm::SmallSetVector<Value, 4> &b) {
  for (Value v : a)
    if (b.contains(v))
      return true;
  return false;
}

// Per-block scan. `pending` enters empty; each `hc.generic` produces
// a barrier-or-not decision and either way updates `pending`. The
// scan recurses into nested regions with a fresh per-block `pending`,
// which is the conservative simplification documented on the pass:
// a write before an `scf.for` does not seed a barrier inside the loop
// body (the outer scan already considers the loop op opaque and
// keeps `pending` live across it; the inner scan starts blank), and
// a write inside one `scf.if` branch does not leak into its sibling.
static void scanBlock(Block &block) {
  OpBuilder builder(block.getParentOp()->getContext());
  llvm::SmallSetVector<Value, 4> pending;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isa<gpu::BarrierOp>(op)) {
      pending.clear();
      continue;
    }
    if (auto generic = dyn_cast<HCGenericOp>(op)) {
      WorkgroupRoots roots = collectWorkgroupRoots(generic);
      bool needsBarrier =
          intersects(roots.reads, pending) || intersects(roots.writes, pending);
      if (needsBarrier) {
        builder.setInsertionPoint(&op);
        gpu::BarrierOp::create(builder, op.getLoc());
        pending.clear();
      }
      for (Value w : roots.writes)
        pending.insert(w);
      continue;
    }
    // Anything else with nested regions (`scf.for`, `scf.if`,
    // `gpu.launch` itself when we walk top-down) — recurse with a
    // fresh per-block scan. We do not peer into the nested ops to
    // discover writes that should join the outer `pending`; the
    // contract is that workgroup-AS writes live in `hc.generic`s
    // and nesting doesn't change that.
    for (Region &region : op.getRegions())
      for (Block &nested : region)
        scanBlock(nested);
  }
}

struct HCInsertWorkgroupBarriersPass
    : public hc::impl::HCInsertWorkgroupBarriersBase<
          HCInsertWorkgroupBarriersPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    root->walk([&](gpu::LaunchOp launch) {
      for (Region &region : launch->getRegions())
        for (Block &block : region)
          scanBlock(block);
    });
  }
};

} // namespace
