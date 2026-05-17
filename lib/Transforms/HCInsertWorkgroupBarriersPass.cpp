// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-insert-workgroup-barriers`. See `doc/lowering.md`
// "Workgroup-AS synchronization" for the contract.

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

// Walk back to the workgroup-AS storage root through single-input
// `unrealized_conversion_cast` (the `ptr<workgroup>` ↔ `bare_tensor`
// sandwich the launch-body type converter plants and that surrounding
// canonicalize/cse has not necessarily folded by this schedule slot)
// and `hc.ptr_offset`. Pointer equality on the returned Value is the
// "same backing allocation" key. Returns {} for non-workgroup-AS operands.
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

// `ins` → reads, `outs` → writes. `outs` is outs-as-init (read at the
// same instant as write — same conservative answer either way).
// Writes kept separate so a precise pass can elide barriers between
// read-only generics on the same root.
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

// Per-block scan with fresh `pending` per nested region. Outer scan
// treats nested ops as opaque; sibling branches don't share writes.
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
    // Recurse into nested regions; workgroup-AS writes only via `hc.generic`.
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
