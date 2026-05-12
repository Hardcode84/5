// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-generic`: unroll-and-merge codegen for
// `hc.generic`. The pass picks an axis order and a partition `p` of
// an unroll budget across the iters, constrained by ixsimpl-provable
// divisibility on each axis bound (no tail loop). For every surviving
// candidate it symbolically generates per-lane offsets and probes
// pairwise contiguity to find maximal contig groups; the
// `(order, partition)` with the highest merge score wins. The body
// stays elementwise scalar at every total unroll: scalar contig
// groups lower to scalar `hc.ptr_load` / `hc.ptr_store`, contig
// groups of size > 1 lower to vector-typed `hc.ptr_load` /
// `hc.ptr_store` of width `G` plumbed through `vector.extract` /
// `vector.from_elements` at the body boundary. The fallback at the
// `(1, ..., 1)` partition matches the scalar baseline emission verb-
// for-verb.
//
// Loop nest shape: outer `scf.parallel` over the parallel iters with
// per-axis step `p_a`; inner `scf.for` nest over the reduction iters
// with `iter_args` carrying one accumulator per (parallel-lane,
// output) pair. Pure-parallel collapses to a single `scf.parallel`,
// pure-reduction to a bare `scf.for` nest.
//
// Per-operand offset expressions surface as `hc.idx_apply` carrying
// the original `ExprAttr` and binding the op's iter syms to the
// loop induction vars (offset by the lane-local delta where the
// partition unrolls); the launch-body lowering downstream walks the
// expression and substitutes the listed operands plus any ambient
// kernel-bound symbols (shape dims, stride params).

#include "hc/Transforms/Passes.h"

#include "LaunchUtils.h"
#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <array>
#include <numeric>

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERGENERIC
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// ---------------------------------------------------------------------------
// Tunables: total unroll budget (`prod(p_i)`) and the axis-order
// permutation cap. Beyond `kAxisOrderCap` the pass uses declaration
// order without permuting (the factorial growth dominates beyond
// rank 4 and per-axis unrolls already cover the common cases).
// `kFactorChoices` enumerates the per-axis factor candidates; powers
// of two only — the merge probe doesn't gain from non-power-of-two
// strides, and the divisibility filter rules out factors that don't
// fit anyway.
constexpr int kUnrollBudget = 32;
constexpr int kAxisOrderCap = 4;
static constexpr std::array<int, 6> kFactorChoices = {1, 2, 4, 8, 16, 32};

// Per-axis info collected from the op. `boundExpr` is set when the
// bound's SSA type is `!hc.idx<expr>`; otherwise the bound is opaque
// (`index`) and the divisibility probe must conservatively reject any
// factor > 1 on this axis.
struct IterAxis {
  size_t origIdx;
  StringRef name;
  IterKind kind;
  Value bound;
  std::optional<sym::ExprHandle> boundExpr;
};

using Partition = SmallVector<int, 4>;
using AxisOrder = SmallVector<size_t, 4>;

// Resolve a probably-comparison node to TRUE/FALSE/UNKNOWN. `ixs_cmp`
// constant-folds to the `IXS_TRUE` / `IXS_FALSE` sentinel when both
// sides reduce to constants — `ixs_check` only inspects bona-fide
// `IXS_CMP` nodes and returns UNKNOWN on a sentinel, so the wrapper
// peeks at the node tag first and falls back to interval propagation
// otherwise. NULL is treated as UNKNOWN (e.g. construction OOM).
static ixs_check_result resolveProbe(ixs_session *session, ixs_node *cmp) {
  if (!cmp)
    return IXS_CHECK_UNKNOWN;
  switch (ixs_node_tag(cmp)) {
  case IXS_TRUE:
    return IXS_CHECK_TRUE;
  case IXS_FALSE:
    return IXS_CHECK_FALSE;
  default:
    return ixs_check(session, cmp, nullptr, 0);
  }
}

// Probe: is `factor` provably a divisor of the axis bound? `factor == 1`
// is trivially true. Without a `boundExpr` (plain `index` bound) the
// probe is conservative and returns false — partitions with `p_a > 1`
// then get filtered out, and the search picks a smaller partition.
static bool probeDivisible(sym::Store &store,
                           const std::optional<sym::ExprHandle> &boundExpr,
                           int factor) {
  if (factor <= 1)
    return true;
  if (!boundExpr)
    return false;
  sym::Session session(store);
  ixs_node *bound = const_cast<ixs_node *>(boundExpr->raw());
  ixs_node *factorNode = ixs_int(session.raw(), factor);
  if (!factorNode)
    return false;
  ixs_node *modNode = ixs_mod(session.raw(), bound, factorNode);
  if (!modNode)
    return false;
  ixs_node *zero = ixs_int(session.raw(), 0);
  if (!zero)
    return false;
  ixs_node *cmp = ixs_cmp(session.raw(), modNode, IXS_CMP_EQ, zero);
  return resolveProbe(session.raw(), cmp) == IXS_CHECK_TRUE;
}

// Probe: does `b - a == 1` hold symbolically? Used pairwise to find
// maximal contig groups in a per-operand offset list. UNKNOWN is
// treated as FALSE — the merge analyzer is conservative; an
// uncertain group stays scalar.
static bool probeContig(sym::Store &store, sym::ExprHandle a,
                        sym::ExprHandle b) {
  sym::Session session(store);
  ixs_node *diff = ixs_sub(session.raw(), const_cast<ixs_node *>(b.raw()),
                           const_cast<ixs_node *>(a.raw()));
  if (!diff)
    return false;
  ixs_node *one = ixs_int(session.raw(), 1);
  if (!one)
    return false;
  ixs_node *cmp = ixs_cmp(session.raw(), diff, IXS_CMP_EQ, one);
  return resolveProbe(session.raw(), cmp) == IXS_CHECK_TRUE;
}

// Build the substitution `s_a -> v_a` (integer literal) over the iter
// syms in `names`, applied to `expr`. Used by the value-typed-outs
// lowering to evaluate offsets to integer slots at compile time —
// `substituteIterDeltas` keeps `s_a` as a free symbol, which is fine
// while the scf.parallel induction var still owns the binding, but
// the value-outs path has no scf.parallel and needs the offset to
// reduce to an integer. Returns `expr` unchanged on OOM / failure.
static sym::ExprHandle substituteIterValues(sym::Store &store,
                                            sym::ExprHandle expr,
                                            ArrayRef<StringRef> names,
                                            ArrayRef<int> vals) {
  if (names.size() != vals.size())
    return expr;
  sym::Session session(store);
  SmallVector<ixs_node *, 4> targets;
  SmallVector<ixs_node *, 4> repls;
  for (auto [n, v] : llvm::zip(names, vals)) {
    SmallString<32> buf(n);
    buf.push_back('\0');
    ixs_node *symNode = ixs_sym(session.raw(), buf.data());
    if (!symNode)
      return expr;
    ixs_node *valNode = ixs_int(session.raw(), v);
    if (!valNode)
      return expr;
    targets.push_back(symNode);
    repls.push_back(valNode);
  }
  if (targets.empty())
    return expr;
  ixs_node *out = ixs_subs_multi(
      session.raw(), const_cast<ixs_node *>(expr.raw()),
      static_cast<uint32_t>(targets.size()), targets.data(), repls.data());
  if (!out)
    return expr;
  return sym::ExprHandle(out);
}

// Build the substitution `s_a -> s_a + delta_a` over the iter syms in
// `names`, applied to `expr`. Used by the merge analyzer to compute
// per-lane offsets symbolically. `delta_a == 0` entries are skipped
// (no-op substitution). Returns `expr` unchanged on OOM / failure.
static sym::ExprHandle substituteIterDeltas(sym::Store &store,
                                            sym::ExprHandle expr,
                                            ArrayRef<StringRef> names,
                                            ArrayRef<int> delta) {
  if (names.size() != delta.size())
    return expr;
  sym::Session session(store);
  SmallVector<ixs_node *, 4> targets;
  SmallVector<ixs_node *, 4> repls;
  for (auto [n, d] : llvm::zip(names, delta)) {
    if (d == 0)
      continue;
    SmallString<32> buf(n);
    buf.push_back('\0');
    ixs_node *symNode = ixs_sym(session.raw(), buf.data());
    if (!symNode)
      return expr;
    ixs_node *deltaNode = ixs_int(session.raw(), d);
    if (!deltaNode)
      return expr;
    ixs_node *shifted = ixs_add(session.raw(), symNode, deltaNode);
    if (!shifted)
      return expr;
    targets.push_back(symNode);
    repls.push_back(shifted);
  }
  if (targets.empty())
    return expr;
  ixs_node *out = ixs_subs_multi(
      session.raw(), const_cast<ixs_node *>(expr.raw()),
      static_cast<uint32_t>(targets.size()), targets.data(), repls.data());
  if (!out)
    return expr;
  return sym::ExprHandle(out);
}

// Decompose lane index `k` (in `[0, prod(p))`) into per-axis deltas
// using axis order `order`: `order[0]` is the most-major axis (varies
// slowest), `order.back()` is the most-minor (varies fastest). Output
// is in declaration order (indexed by `axes`) so callers can read
// `delta[a]` directly without translating through `order`.
static SmallVector<int, 4> decomposeLane(int k, ArrayRef<size_t> order,
                                         ArrayRef<int> p) {
  SmallVector<int, 4> delta(p.size(), 0);
  for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
    size_t a = order[i];
    int pa = p[a];
    delta[a] = k % pa;
    k /= pa;
  }
  return delta;
}

// Maximal contig group in an offset list. Each `(start, size)` covers
// lanes `[start, start + size)` whose pairwise offset diffs all
// resolve to TRUE under the contig probe. Group sizes always include
// at least one lane; size > 1 is the merge-eligible case.
struct ContigGroup {
  int start;
  int size;
};

static SmallVector<ContigGroup>
findContigGroups(sym::Store &store, ArrayRef<sym::ExprHandle> offsets) {
  SmallVector<ContigGroup> groups;
  if (offsets.empty())
    return groups;
  int n = static_cast<int>(offsets.size());
  int gStart = 0;
  for (int k = 1; k < n; ++k) {
    if (!probeContig(store, offsets[k - 1], offsets[k])) {
      groups.push_back({gStart, k - gStart});
      gStart = k;
    }
  }
  groups.push_back({gStart, n - gStart});
  return groups;
}

// Enumerate axis orders. For `n <= kAxisOrderCap` the search yields
// every permutation of `[0, n)`; for larger ranks only the
// declaration-order identity is considered.
static SmallVector<AxisOrder> enumerateAxisOrders(int n) {
  AxisOrder ident(n);
  std::iota(ident.begin(), ident.end(), size_t{0});
  if (n > kAxisOrderCap)
    return {ident};
  SmallVector<AxisOrder> out;
  AxisOrder cur = ident;
  do {
    out.push_back(cur);
  } while (std::next_permutation(cur.begin(), cur.end()));
  return out;
}

// Enumerate partitions of `kUnrollBudget` over `n` axes. Each per-axis
// factor comes from `kFactorChoices`; the running product is bounded
// by `budget`. Output includes the trivial `(1, ..., 1)` partition.
static SmallVector<Partition> enumeratePartitions(int n, int budget) {
  SmallVector<Partition> out;
  if (n <= 0) {
    out.push_back(Partition{});
    return out;
  }
  Partition cur(n, 1);
  std::function<void(int, int)> rec = [&](int axis, int rem) {
    if (axis == n) {
      out.push_back(cur);
      return;
    }
    for (int f : kFactorChoices) {
      if (f > rem)
        break;
      cur[axis] = f;
      rec(axis + 1, rem / f);
    }
  };
  rec(0, budget);
  return out;
}

// Collect the per-axis info from the op. The bound's SSA type is
// inspected for an `!hc.idx<expr>` payload — that's where the
// divisibility probe gets a symbolic axis size from. Plain `index`
// bounds yield an absent `boundExpr`.
static SmallVector<IterAxis> collectIterAxes(HCGenericOp op) {
  SmallVector<IterAxis> axes;
  ArrayAttr symsAttr = op.getIterSymsAttr();
  ArrayAttr kindsAttr = op.getIterKindsAttr();
  ValueRange bounds = op.getIterBounds();
  axes.reserve(symsAttr.size());
  for (size_t i = 0; i < symsAttr.size(); ++i) {
    IterAxis ax;
    ax.origIdx = i;
    ax.name = cast<StringAttr>(symsAttr[i]).getValue();
    ax.kind = cast<IterKindAttr>(kindsAttr[i]).getValue();
    ax.bound = bounds[i];
    if (auto idxTy = dyn_cast<IdxType>(ax.bound.getType()))
      if (ExprAttr e = idxTy.getExpr())
        ax.boundExpr = e.getValue();
    axes.push_back(ax);
  }
  return axes;
}

// Total partition product: how many body clones a candidate produces
// per innermost loop iteration.
static int prodOf(ArrayRef<int> p) {
  int prod = 1;
  for (int v : p)
    prod *= v;
  return prod;
}

// Subset prod over axes whose iter kind matches `kind`.
static int prodOfKind(ArrayRef<IterAxis> axes, ArrayRef<int> p, IterKind kind) {
  int prod = 1;
  for (auto [ax, pa] : llvm::zip(axes, p))
    if (ax.kind == kind)
      prod *= pa;
  return prod;
}

// Assemble per-lane offsets for one operand, given a global axis order
// `order` and partition `p`. `mask` selects which axes to vary —
// inputs use every axis (full `prod(p)` lanes), outputs use only
// parallel axes (`prodPar` lanes; the verifier already guarantees the
// outs offset is independent of reduction syms). Lanes are decomposed
// in `order`'s rightmost-fastest convention against the surviving
// (masked) axes only.
static SmallVector<sym::ExprHandle>
laneOffsets(sym::Store &store, ExprAttr origOffset, ArrayRef<IterAxis> axes,
            ArrayRef<size_t> order, ArrayRef<int> p, ArrayRef<bool> includeAxis,
            int laneCount) {
  SmallVector<sym::ExprHandle> result;
  result.reserve(laneCount);
  SmallVector<size_t, 4> filteredOrder;
  for (size_t a : order)
    if (includeAxis[a])
      filteredOrder.push_back(a);
  SmallVector<int, 4> filteredP;
  filteredP.reserve(filteredOrder.size());
  for (size_t a : filteredOrder)
    filteredP.push_back(p[a]);
  SmallVector<StringRef, 4> names;
  names.reserve(axes.size());
  for (const IterAxis &ax : axes)
    names.push_back(ax.name);
  for (int lane = 0; lane < laneCount; ++lane) {
    SmallVector<int, 4> deltaFiltered =
        decomposeLane(lane, filteredOrder, filteredP);
    SmallVector<int, 4> delta(axes.size(), 0);
    for (auto [a, d] : llvm::zip(filteredOrder, deltaFiltered))
      delta[a] = d;
    result.push_back(
        substituteIterDeltas(store, origOffset.getValue(), names, delta));
  }
  return result;
}

// Score a candidate `(order, partition)` by its merge potential:
// per-operand contig-group analysis, sum of `(group_size - 1)` over
// every group in every operand. A bigger total means more loads/
// stores collapsed at the boundary. The trivial partition scores 0.
// Pure-zero scores from non-trivial partitions still beat the
// trivial one only when no smaller partition merges either; ties
// resolve via the lex-order tiebreaker in the caller.
static int scoreCandidate(HCGenericOp op, ArrayRef<IterAxis> axes,
                          ArrayRef<size_t> order, ArrayRef<int> p,
                          sym::Store &store) {
  int prod = prodOf(p);
  int prodPar = prodOfKind(axes, p, IterKind::Parallel);
  SmallVector<bool, 4> insMask(axes.size(), true);
  SmallVector<bool, 4> outsMask(axes.size(), false);
  for (size_t i = 0; i < axes.size(); ++i)
    if (axes[i].kind == IterKind::Parallel)
      outsMask[i] = true;

  int score = 0;
  ArrayAttr insOff = op.getInsOffsetsAttr();
  for (auto [i, in] : llvm::enumerate(op.getIns())) {
    auto perOp = cast<ArrayAttr>(insOff[i]);
    if (perOp.size() != 1)
      return 0;
    auto off = cast<ExprAttr>(perOp[0]);
    auto offsets = laneOffsets(store, off, axes, order, p, insMask, prod);
    for (const ContigGroup &g : findContigGroups(store, offsets))
      score += g.size - 1;
    (void)in;
  }
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (auto [i, out] : llvm::enumerate(op.getOuts())) {
    auto perOp = cast<ArrayAttr>(outsOff[i]);
    if (perOp.size() != 1)
      return 0;
    auto off = cast<ExprAttr>(perOp[0]);
    auto offsets = laneOffsets(store, off, axes, order, p, outsMask, prodPar);
    // Outs init load + final store both ride this contig analysis;
    // each merge eliminates two boundary ops, so the score weights
    // accordingly.
    for (const ContigGroup &g : findContigGroups(store, offsets))
      score += 2 * (g.size - 1);
    (void)out;
  }
  return score;
}

// Pick `(order, partition)` with the highest merge score. Filters out
// candidates that violate the divisibility constraint on any axis
// (per-axis factor must provably divide its bound). Ties resolve
// lexicographically on `order` and then on `p`. Falls back to the
// trivial partition `(1, ..., 1)` when no non-trivial candidate
// survives — that case matches the scalar baseline emission.
static std::pair<AxisOrder, Partition>
selectBest(HCGenericOp op, ArrayRef<IterAxis> axes, sym::Store &store) {
  int n = static_cast<int>(axes.size());
  AxisOrder bestOrder(n);
  std::iota(bestOrder.begin(), bestOrder.end(), size_t{0});
  Partition bestP(n, 1);
  int bestScore = -1;
  for (const AxisOrder &order : enumerateAxisOrders(n)) {
    for (const Partition &p : enumeratePartitions(n, kUnrollBudget)) {
      bool ok = true;
      for (size_t a = 0; a < axes.size(); ++a) {
        if (!probeDivisible(store, axes[a].boundExpr, p[a])) {
          ok = false;
          break;
        }
      }
      if (!ok)
        continue;
      int score = scoreCandidate(op, axes, order, p, store);
      if (score > bestScore) {
        bestScore = score;
        bestOrder = order;
        bestP = p;
      }
    }
  }
  return {bestOrder, bestP};
}

// Materialize a Value usable from index-consuming ops. Builtin
// `index` values pass through; `!hc.idx<...>` SSA crosses via
// `unrealized_conversion_cast`, which the launch-body lowering /
// reconcile-unrealized-casts collapse downstream once both ends are
// concrete.
static Value castIdxToIndex(OpBuilder &builder, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), v)
      .getResult(0);
}

// Returns the static rank-1 element count for a value-typed `outs`
// carrier — `BareVectorType` / `BareTensorType` shape entry resolved
// to an integer literal, or a builtin `VectorType` direct dim. Used
// to size the `vector.from_elements` compose at the parallel-sweep
// boundary. Symbolic shapes (`["N"]` with `N` ambient) are rejected:
// the compose has to know the lane count at compile time.
static std::optional<int64_t> getValueOutLaneCount(Type t) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(t)) {
    ShapeAttr shape = shaped.getSymbolicShape();
    if (!shape || shape.getDims().size() != 1)
      return std::nullopt;
    auto expr = dyn_cast<ExprAttr>(shape.getDims().front());
    if (!expr)
      return std::nullopt;
    return sym::getIntegerLiteralValue(expr.getValue());
  }
  if (auto v = dyn_cast<mlir::VectorType>(t)) {
    if (v.getRank() != 1 || v.isScalable())
      return std::nullopt;
    return v.getShape()[0];
  }
  return std::nullopt;
}

// True when this op has at least one value-typed (non-ptr) outs
// operand. Drives the dispatch in `lowerOne`: value-typed outs need
// a compile-time-unrolled compose (no `scf.parallel`) so the result
// can live in a single SSA register; the ptr-only path still uses
// the partition-aware `scf.parallel` shape.
static bool hasValueOuts(HCGenericOp op) {
  for (Value v : op.getOuts())
    if (!isa<PtrType>(v.getType()))
      return true;
  return false;
}

// Pre-flight check: returns `true` when this op matches the v0
// codegen scope. The ptr-only path accepts any rank-1 offset with
// resolved iter bounds. The value-outs path additionally requires
// constant iter bounds (full unroll happens at compile time), all
// iters parallel (reduction-with-value-out is a v1 follow-up — the
// accumulator threading across an `scf.for` doesn't compose with
// the single-SSA-result boundary), a plain `hc.yield` terminator
// (predicated-yield against a value-out is a v1 follow-up too), and
// every outs offset's free syms confined to iter syms — anything
// else makes the slot index ambient-dependent and the compile-time
// slot evaluation can't pin it. Mismatch leaves the op in place —
// we'd rather skip than partially lower. Opaque pointers (no
// element type on the carrier) also bail: the v0 load emission
// needs the element type to spell the result, and the pointer's
// `$elementType` is the only available source.
static bool isV0Candidate(HCGenericOp op) {
  auto goodPtr = [](Value v) {
    auto p = dyn_cast<PtrType>(v.getType());
    return p && p.getElementType();
  };
  for (Value v : op.getIns())
    if (!goodPtr(v))
      return false;
  bool hasValOut = false;
  for (Value v : op.getOuts()) {
    if (goodPtr(v))
      continue;
    if (!getValueOutLaneCount(v.getType()))
      return false;
    hasValOut = true;
  }
  ArrayAttr insOff = op.getInsOffsetsAttr();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (Attribute perOp : insOff)
    if (cast<ArrayAttr>(perOp).size() != 1)
      return false;
  for (Attribute perOp : outsOff)
    if (cast<ArrayAttr>(perOp).size() != 1)
      return false;
  for (Value bound : op.getIterBounds())
    if (auto def = bound.getDefiningOp())
      if (isa<HCUndefValueOp>(def))
        return false;
  for (Operation &nested : op.getBody().front())
    if (isa<scf::IfOp>(nested))
      return false;
  if (!hasValOut)
    return true;

  // Value-outs-specific extra conditions. Iter bounds: each one must
  // resolve to a compile-time integer (either an `arith.constant`
  // index value or an `!hc.idx<"<int>">`-typed SSA bound — the
  // frontend planting `hc.const` against an integer-only shape is
  // the canonical case).
  llvm::StringSet<> iterNames;
  for (Attribute s : op.getIterSymsAttr())
    iterNames.insert(cast<StringAttr>(s).getValue());
  for (Value bound : op.getIterBounds()) {
    std::optional<int64_t> v;
    if (auto def = bound.getDefiningOp<arith::ConstantOp>())
      if (auto attr = dyn_cast<IntegerAttr>(def.getValue()))
        v = attr.getInt();
    if (!v) {
      if (auto idxTy = dyn_cast<IdxType>(bound.getType()))
        if (ExprAttr e = idxTy.getExpr())
          v = sym::getIntegerLiteralValue(e.getValue());
    }
    if (!v || *v < 0)
      return false;
  }
  // All iters must be parallel — the value-outs compose threads
  // every parLane through the result vector, and a reduction iter
  // would need cross-lane carry that the boundary form doesn't
  // model in v0.
  for (Attribute k : op.getIterKindsAttr())
    if (cast<IterKindAttr>(k).getValue() != IterKind::Parallel)
      return false;
  // Body terminator must be `hc.yield` — predicated-yield against
  // a value-out needs select-at-the-boundary, deferred to v1.
  if (!isa<HCYieldOp>(op.getBody().front().back()))
    return false;
  // Outs offset's free syms must be a subset of iter syms; ambient
  // syms (shape / stride params) would make slot evaluation
  // ambient-dependent.
  for (Attribute perOp : outsOff) {
    auto off = cast<ExprAttr>(cast<ArrayAttr>(perOp)[0]);
    bool ok = true;
    sym::walkSymbolNames(off.getValue(), [&](StringRef name) {
      if (!iterNames.contains(name))
        ok = false;
    });
    if (!ok)
      return false;
  }
  return true;
}

// Pull the single composed offset expression for an operand at
// position `idx` from an `ins_offsets` / `outs_offsets` array.
// Caller has already checked rank-1 via `isV0Candidate`.
static ExprAttr getOperandOffset(ArrayAttr arrayAttr, size_t idx) {
  return cast<ExprAttr>(cast<ArrayAttr>(arrayAttr[idx])[0]);
}

// Seed a sym-name → SSA binding map from the op's captured ambient
// bindings. `hc-flatten-with-layouts` walks the kernel-arg bundle /
// launch context while every ambient sym is reachable as an HC-typed
// SSA and stamps the (sym, value) pairs onto `ambient_idxs` /
// `ambient_idx_syms`. Lower-generic consumes them through this seed
// so the per-lane offset emission inherits the dataflow edge instead
// of re-discovering each ambient name via an ancestor walk at planting
// time. The map's later writes (iter syms, partition coords) override
// any name they collide with — iter / partition bindings are scoped
// to the per-lane scf body and take precedence over ambient SSAs
// that happen to share a name with a body sym.
static void seedAmbientScope(HCGenericOp op, llvm::StringMap<Value> &scope) {
  ArrayAttr syms = op.getAmbientIdxSymsAttr();
  OperandRange vals = op.getAmbientIdxs();
  for (auto [val, symAttr] :
       llvm::zip_equal(vals, syms.getAsRange<StringAttr>()))
    scope.try_emplace(symAttr.getValue(), val);
}

// Materialize one offset expression as `index`-typed SSA. Filters
// the loop-scope binding map down to the symbols that actually
// occur in the offset expression — `hc.idx_apply`'s verifier
// requires every listed symbol to be free in the expression — then
// emits the apply and casts the resulting `!hc.idx<expr>` to
// `index` for the consumer's `hc.ptr_offset`.
//
// `loopScope` is seeded with the op's `ambient_idxs` (kernel-arg
// dim / stride syms, launch geometry, structured-loop join syms —
// captured at flatten time) plus the per-iteration loop induction
// vars and partition coordinates. Any symbol the caller didn't pre-
// bind stays free in the planted apply and falls through to ambient-
// context resolution downstream; today the only such names are the
// ones flatten couldn't locate in scope, which the kernel-arg-bundle
// resolver still handles. Once every emission path is ambient-bound
// the fallback becomes unreachable.
static Value emitOffset(OpBuilder &builder, Location loc, ExprAttr offsetExpr,
                        const llvm::StringMap<Value> &loopScope) {
  llvm::StringSet<> freeSyms;
  sym::walkSymbolNames(offsetExpr.getValue(),
                       [&](StringRef name) { freeSyms.insert(name); });

  // Sort the binding order lexicographically: `llvm::StringMap`
  // iteration is unordered, and `hc.idx_apply`'s textual form pins
  // the operand list, so leaving it to map order would make
  // round-trip tests fragile across rebuilds.
  SmallVector<StringRef> names;
  for (const auto &entry : loopScope)
    if (freeSyms.contains(entry.getKey()))
      names.push_back(entry.getKey());
  llvm::sort(names);

  SmallVector<Attribute> symAttrs;
  SmallVector<Value> operands;
  for (StringRef name : names) {
    symAttrs.push_back(builder.getStringAttr(name));
    operands.push_back(loopScope.lookup(name));
  }

  auto idxTy = IdxType::get(builder.getContext(), offsetExpr);
  Value applied = HCIdxApplyOp::create(builder, loc, idxTy, operands,
                                       builder.getArrayAttr(symAttrs))
                      .getResult();
  return castIdxToIndex(builder, loc, applied);
}

// Body cloner. Replaces block-arg references with their loaded /
// carry values via the IR mapping, drops the `hc.yield` (its values
// are surfaced through `yieldedOut` for the caller to feed into the
// SCF loop's terminator), and emits the cloned ops at the builder's
// current insertion point.
static LogicalResult cloneBody(OpBuilder &builder, HCGenericOp op,
                               ValueRange insVals, ValueRange outsVals,
                               SmallVectorImpl<Value> &yieldedOut) {
  Block &src = op.getBody().front();
  IRMapping mapping;
  size_t pos = 0;
  for (Value v : insVals)
    mapping.map(src.getArgument(pos++), v);
  for (Value v : outsVals)
    mapping.map(src.getArgument(pos++), v);
  auto yield = dyn_cast<HCYieldOp>(src.back());
  if (!yield)
    return op.emitOpError("body must end with `hc.yield`");
  for (Operation &nested : src) {
    if (&nested == yield.getOperation())
      break;
    builder.clone(nested, mapping);
  }
  yieldedOut.clear();
  yieldedOut.reserve(yield.getValues().size());
  for (Value v : yield.getValues())
    yieldedOut.push_back(mapping.lookupOrDefault(v));
  return success();
}

// Materialize one lane's offset as `index`-typed SSA via
// `hc.idx_apply`. The substituted expression carries the per-axis
// `delta` baked in (each iter sym `s_a` shifted to `s_a + delta_a`),
// so the apply binds the same loop induction vars as the trivial
// case while spelling the lane-local offset; for `delta == 0` across
// every axis the substitution is a no-op and emission collapses to
// the scalar baseline verbatim.
static Value emitLaneOffset(OpBuilder &builder, Location loc,
                            ExprAttr origOffset,
                            const llvm::StringMap<Value> &scope,
                            sym::Store &store, ArrayRef<IterAxis> axes,
                            ArrayRef<size_t> order, ArrayRef<int> p,
                            ArrayRef<bool> includeAxis, int lane) {
  SmallVector<size_t, 4> filteredOrder;
  for (size_t a : order)
    if (includeAxis[a])
      filteredOrder.push_back(a);
  SmallVector<int, 4> filteredP;
  filteredP.reserve(filteredOrder.size());
  for (size_t a : filteredOrder)
    filteredP.push_back(p[a]);
  SmallVector<int, 4> deltaFiltered =
      decomposeLane(lane, filteredOrder, filteredP);
  SmallVector<int, 4> delta(axes.size(), 0);
  for (auto [a, d] : llvm::zip(filteredOrder, deltaFiltered))
    delta[a] = d;
  SmallVector<StringRef, 4> names;
  names.reserve(axes.size());
  for (const IterAxis &ax : axes)
    names.push_back(ax.name);
  sym::ExprHandle laneExpr =
      substituteIterDeltas(store, origOffset.getValue(), names, delta);
  ExprAttr laneAttr = ExprAttr::get(builder.getContext(), laneExpr);
  return emitOffset(builder, loc, laneAttr, scope);
}

// Fold a per-axis delta tuple back into a flat lane index restricted
// to `subOrder` (a subsequence of `order`, in same direction). Used
// to map a full lane (covering all axes) to its parallel-axis
// projection — the slot in the iter_args carry where its accumulator
// lives.
static int computeSubLane(ArrayRef<int> delta, ArrayRef<size_t> subOrder,
                          ArrayRef<int> p) {
  int sub = 0;
  int stride = 1;
  for (int i = static_cast<int>(subOrder.size()) - 1; i >= 0; --i) {
    size_t a = subOrder[i];
    sub += delta[a] * stride;
    stride *= p[a];
  }
  return sub;
}

// Emit a contig group's load (scalar for `size == 1`, vector
// `<G x T>` otherwise) and write the per-lane scalars into `out` at
// indices `[start, start + size)`. Vector loads route through
// `vector.extract` to recover lane-scoped scalars feeding the body.
static void emitGroupLoad(OpBuilder &builder, Location loc, Value ptr,
                          Type elemTy, Value baseAddr, const ContigGroup &g,
                          MutableArrayRef<Value> out) {
  if (g.size == 1) {
    Value v = HCPtrLoadOp::create(builder, loc, elemTy, baseAddr).getResult();
    out[g.start] = v;
    return;
  }
  auto vecTy = mlir::VectorType::get({g.size}, elemTy);
  Value vec = HCPtrLoadOp::create(builder, loc, vecTy, baseAddr).getResult();
  for (int k = 0; k < g.size; ++k) {
    Value lane =
        vector::ExtractOp::create(builder, loc, vec, ArrayRef<int64_t>{k})
            .getResult();
    out[g.start + k] = lane;
  }
}

// Emit a contig group's store (scalar for `size == 1`, vector
// `<G x T>` otherwise). Vector stores pack the per-lane scalars into
// a vector via `vector.from_elements` at the boundary.
static void emitGroupStore(OpBuilder &builder, Location loc, Type elemTy,
                           Value baseAddr, const ContigGroup &g,
                           ArrayRef<Value> laneVals) {
  if (g.size == 1) {
    HCPtrStoreOp::create(builder, loc, laneVals[g.start], baseAddr);
    return;
  }
  SmallVector<Value> elems;
  elems.reserve(g.size);
  for (int k = 0; k < g.size; ++k)
    elems.push_back(laneVals[g.start + k]);
  auto vecTy = mlir::VectorType::get({g.size}, elemTy);
  Value vec =
      vector::FromElementsOp::create(builder, loc, vecTy, elems).getResult();
  HCPtrStoreOp::create(builder, loc, vec, baseAddr);
}

// Emit per-lane loads for every input of `op`, returning a flat
// `[in_idx][lane]` 2D buffer of SSA values. Per-input contig groups
// drive load shape: scalar groups emit single `hc.ptr_load`, groups
// of size > 1 emit vector loads + extracts. Lanes are decomposed
// against every iter axis (mask = all-true).
static SmallVector<SmallVector<Value>>
emitInsLoadsLaned(OpBuilder &builder, HCGenericOp op, ArrayRef<IterAxis> axes,
                  ArrayRef<size_t> order, ArrayRef<int> p,
                  const llvm::StringMap<Value> &scope, sym::Store &store) {
  Location loc = op.getLoc();
  int prodAll = prodOf(p);
  size_t numIns = op.getIns().size();
  SmallVector<SmallVector<Value>> result(numIns);
  ArrayAttr insOff = op.getInsOffsetsAttr();
  SmallVector<bool, 4> allMask(axes.size(), true);
  for (size_t ii = 0; ii < numIns; ++ii) {
    ExprAttr origOff = getOperandOffset(insOff, ii);
    SmallVector<sym::ExprHandle> offs =
        laneOffsets(store, origOff, axes, order, p, allMask, prodAll);
    auto groups = findContigGroups(store, offs);
    Value ptr = op.getIns()[ii];
    Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
    result[ii].resize(prodAll);
    for (const ContigGroup &g : groups) {
      Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes,
                                 order, p, allMask, g.start);
      Value addr = HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                       .getResult();
      emitGroupLoad(builder, loc, ptr, elemTy, addr, g, result[ii]);
    }
  }
  return result;
}

// Emit per-output outs init loads at parallel-iter scope. Lanes
// decompose only over the parallel axis subset (mask = parallel
// axes), giving `prodPar` slots per output. Returned flat layout is
// `[par_lane * numOuts + out_idx]` so it slots directly into
// `scf.for`'s `iter_args`.
static SmallVector<Value> emitOutsInitLoadsPartitioned(
    OpBuilder &builder, HCGenericOp op, ArrayRef<IterAxis> axes,
    ArrayRef<size_t> order, ArrayRef<int> p,
    const llvm::StringMap<Value> &scope, sym::Store &store) {
  Location loc = op.getLoc();
  int prodPar = prodOfKind(axes, p, IterKind::Parallel);
  size_t numOuts = op.getOuts().size();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  SmallVector<bool, 4> parMask(axes.size(), false);
  for (size_t a = 0; a < axes.size(); ++a)
    if (axes[a].kind == IterKind::Parallel)
      parMask[a] = true;
  SmallVector<SmallVector<Value>> perOut(numOuts);
  for (size_t oi = 0; oi < numOuts; ++oi) {
    ExprAttr origOff = getOperandOffset(outsOff, oi);
    SmallVector<sym::ExprHandle> offs =
        laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
    auto groups = findContigGroups(store, offs);
    Value ptr = op.getOuts()[oi];
    Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
    perOut[oi].resize(prodPar);
    for (const ContigGroup &g : groups) {
      Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes,
                                 order, p, parMask, g.start);
      Value addr = HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                       .getResult();
      emitGroupLoad(builder, loc, ptr, elemTy, addr, g, perOut[oi]);
    }
  }
  SmallVector<Value> flat;
  flat.reserve(prodPar * numOuts);
  for (int pl = 0; pl < prodPar; ++pl)
    for (size_t oi = 0; oi < numOuts; ++oi)
      flat.push_back(perOut[oi][pl]);
  return flat;
}

// Emit per-output outs stores at parallel-iter scope. Mirrors the
// init-load path: contig groups of size > 1 pack per-lane scalars
// via `vector.from_elements`, scalar groups emit one `hc.ptr_store`.
static void emitOutsStoresPartitioned(OpBuilder &builder, HCGenericOp op,
                                      ArrayRef<IterAxis> axes,
                                      ArrayRef<size_t> order, ArrayRef<int> p,
                                      ValueRange flatFinals,
                                      const llvm::StringMap<Value> &scope,
                                      sym::Store &store) {
  Location loc = op.getLoc();
  int prodPar = prodOfKind(axes, p, IterKind::Parallel);
  size_t numOuts = op.getOuts().size();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  SmallVector<bool, 4> parMask(axes.size(), false);
  for (size_t a = 0; a < axes.size(); ++a)
    if (axes[a].kind == IterKind::Parallel)
      parMask[a] = true;
  for (size_t oi = 0; oi < numOuts; ++oi) {
    ExprAttr origOff = getOperandOffset(outsOff, oi);
    SmallVector<sym::ExprHandle> offs =
        laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
    auto groups = findContigGroups(store, offs);
    Value ptr = op.getOuts()[oi];
    Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
    SmallVector<Value> laneVals(prodPar);
    for (int pl = 0; pl < prodPar; ++pl)
      laneVals[pl] = flatFinals[pl * numOuts + oi];
    for (const ContigGroup &g : groups) {
      Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes,
                                 order, p, parMask, g.start);
      Value addr = HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                       .getResult();
      emitGroupStore(builder, loc, elemTy, addr, g, laneVals);
    }
  }
}

// Innermost body emission: loads ins for every lane, clones the body
// `prod(p)` times in axis-order lex (rightmost varies fastest), and
// threads the `prodPar * numOuts` accumulator through. Within one
// invocation the body sees scalar ins (its lane's loaded value) and
// scalar outs (the running accumulator at this body's parallel-lane
// slot); the post-body `acc[parLane] = yielded` assignment makes the
// reduction-axis unrolls compose into the same accumulator slot in
// declaration order.
static FailureOr<SmallVector<Value>>
emitInnerBodyClones(OpBuilder &builder, HCGenericOp op, ArrayRef<IterAxis> axes,
                    ArrayRef<size_t> order, ArrayRef<int> p,
                    ValueRange iterArgs, const llvm::StringMap<Value> &scope,
                    sym::Store &store) {
  int prodAll = prodOf(p);
  int prodPar = prodOfKind(axes, p, IterKind::Parallel);
  size_t numOuts = op.getOuts().size();
  size_t numIns = op.getIns().size();

  SmallVector<size_t, 4> parOrder;
  for (size_t a : order)
    if (axes[a].kind == IterKind::Parallel)
      parOrder.push_back(a);

  SmallVector<SmallVector<Value>> acc(prodPar, SmallVector<Value>(numOuts));
  for (int pl = 0; pl < prodPar; ++pl)
    for (size_t oi = 0; oi < numOuts; ++oi)
      acc[pl][oi] = iterArgs[pl * numOuts + oi];

  SmallVector<SmallVector<Value>> insLanes =
      emitInsLoadsLaned(builder, op, axes, order, p, scope, store);

  for (int lane = 0; lane < prodAll; ++lane) {
    SmallVector<int, 4> delta = decomposeLane(lane, order, p);
    int parLane = computeSubLane(delta, parOrder, p);
    SmallVector<Value> insVals(numIns);
    for (size_t ii = 0; ii < numIns; ++ii)
      insVals[ii] = insLanes[ii][lane];
    SmallVector<Value> outsVals = acc[parLane];
    SmallVector<Value> yielded;
    if (failed(cloneBody(builder, op, insVals, outsVals, yielded)))
      return failure();
    if (yielded.size() != numOuts)
      return op.emitOpError("body yielded wrong arity");
    acc[parLane] = std::move(yielded);
  }

  SmallVector<Value> flat;
  flat.reserve(prodPar * numOuts);
  for (int pl = 0; pl < prodPar; ++pl)
    for (size_t oi = 0; oi < numOuts; ++oi)
      flat.push_back(acc[pl][oi]);
  return flat;
}

// Reduction nest with per-axis step `p[a]`. Each `scf.for` carries
// the same `prodPar * numOuts` flat accumulator through `iter_args`;
// the innermost level invokes `emitInnerBodyClones` to expand the
// `prod(p)` body invocations and yields the updated accumulator.
static FailureOr<SmallVector<Value>> emitReductionNestPartitioned(
    OpBuilder &builder, HCGenericOp op, ArrayRef<IterAxis> axes,
    ArrayRef<size_t> order, ArrayRef<int> p, ArrayRef<size_t> redOrder,
    ValueRange initAccs, llvm::StringMap<Value> scope, sym::Store &store) {
  Location loc = op.getLoc();
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  ValueRange bounds = op.getIterBounds();
  std::function<FailureOr<SmallVector<Value>>(size_t, ValueRange,
                                              llvm::StringMap<Value> &)>
      build = [&](size_t depth, ValueRange iterArgs,
                  llvm::StringMap<Value> &localScope)
      -> FailureOr<SmallVector<Value>> {
    if (depth == redOrder.size())
      return emitInnerBodyClones(builder, op, axes, order, p, iterArgs,
                                 localScope, store);
    size_t a = redOrder[depth];
    StringRef name = axes[a].name;
    Value bound = castIdxToIndex(builder, loc, bounds[a]);
    Value step = arith::ConstantIndexOp::create(builder, loc, p[a]).getResult();
    auto forOp = scf::ForOp::create(builder, loc, c0, bound, step, iterArgs);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    auto prev = localScope.lookup(name);
    localScope[name] = forOp.getInductionVar();
    auto inner = build(depth + 1, forOp.getRegionIterArgs(), localScope);
    if (prev)
      localScope[name] = prev;
    else
      localScope.erase(name);
    if (failed(inner))
      return failure();
    scf::YieldOp::create(builder, loc, *inner);
    return SmallVector<Value>(forOp.getResults().begin(),
                              forOp.getResults().end());
  };
  return build(0, initAccs, scope);
}

// Top-level emit: outer `scf.parallel` over the parallel iters with
// per-axis step `p[a]`, an inner reduction nest at `p[a]` step per
// reduction axis, body cloned `prod(p)` times per innermost
// iteration. The iter_args carry one accumulator per (parallel-lane,
// output) pair; `emitOutsInitLoadsPartitioned` and
// `emitOutsStoresPartitioned` handle the boundary loads / stores at
// parallel-iter scope, with contig merging where the analyzer found
// merge-eligible groups.
static LogicalResult lowerWithPartition(HCGenericOp op, ArrayRef<IterAxis> axes,
                                        ArrayRef<size_t> order,
                                        ArrayRef<int> p) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

  SmallVector<size_t> parIdx, redIdx;
  for (size_t i = 0; i < axes.size(); ++i)
    (axes[i].kind == IterKind::Parallel ? parIdx : redIdx).push_back(i);

  SmallVector<size_t, 4> redOrder;
  for (size_t a : order)
    if (axes[a].kind == IterKind::Reduction)
      redOrder.push_back(a);

  ValueRange bounds = op.getIterBounds();

  auto buildPerParallel = [&](OpBuilder &b,
                              llvm::StringMap<Value> &scope) -> LogicalResult {
    SmallVector<Value> initOuts =
        emitOutsInitLoadsPartitioned(b, op, axes, order, p, scope, store);
    SmallVector<Value> finals;
    if (redOrder.empty()) {
      auto yielded =
          emitInnerBodyClones(b, op, axes, order, p, initOuts, scope, store);
      if (failed(yielded))
        return failure();
      finals = std::move(*yielded);
    } else {
      auto reduced = emitReductionNestPartitioned(
          b, op, axes, order, p, redOrder, initOuts, scope, store);
      if (failed(reduced))
        return failure();
      finals = std::move(*reduced);
    }
    emitOutsStoresPartitioned(b, op, axes, order, p, finals, scope, store);
    return success();
  };

  if (parIdx.empty()) {
    llvm::StringMap<Value> scope;
    seedAmbientScope(op, scope);
    return buildPerParallel(builder, scope);
  }

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  SmallVector<Value> lowers(parIdx.size(), c0);
  SmallVector<Value> uppers;
  SmallVector<Value> steps;
  uppers.reserve(parIdx.size());
  steps.reserve(parIdx.size());
  for (size_t pi : parIdx) {
    uppers.push_back(castIdxToIndex(builder, loc, bounds[pi]));
    steps.push_back(
        arith::ConstantIndexOp::create(builder, loc, p[pi]).getResult());
  }

  LogicalResult bodyStatus = success();
  scf::ParallelOp::create(builder, loc, lowers, uppers, steps,
                          [&](OpBuilder &b, Location, ValueRange ivs) {
                            llvm::StringMap<Value> scope;
                            seedAmbientScope(op, scope);
                            for (auto [k, pi] : llvm::enumerate(parIdx))
                              scope[axes[pi].name] = ivs[k];
                            bodyStatus = buildPerParallel(b, scope);
                          });
  return bodyStatus;
}

// Collective dispatch detector: at least one outs operand is
// `!hc.ptr<workgroup, T>` (LDS-staged tile), every iter is parallel
// (no cross-thread accumulation), and the op sits inside a
// `gpu.launch` (we need the dim3 thread/block layout to chunk the
// iter space across the wave). Per-lane outs falls through to the
// existing partition-aware path — running scf.parallel without
// partitioning is the correct shape for lane-local results.
//
// The workgroup-ptr signal is the dispositive one: a workgroup tile
// is shared state, so emitting "every lane runs every iteration"
// against it would have the wave fight itself for every element.
// All-parallel + launch-enclosed are necessary follow-ups: cross-iter
// reduction needs cross-thread synchronization the collective shape
// doesn't model, and the chunk loop's `lin_tid` source disappears
// outside a launch.
static bool isCollectiveCandidate(HCGenericOp op, ArrayRef<IterAxis> axes) {
  bool hasWorkgroupOut = false;
  for (Value v : op.getOuts()) {
    auto ptr = dyn_cast<PtrType>(v.getType());
    if (!ptr)
      return false;
    if (ptr.getAddrSpace() == AddrSpace::Workgroup)
      hasWorkgroupOut = true;
  }
  if (!hasWorkgroupOut)
    return false;
  for (const IterAxis &ax : axes)
    if (ax.kind != IterKind::Parallel)
      return false;
  return op->getParentOfType<gpu::LaunchOp>() != nullptr;
}

// Collective dispatch emit. Each thread of the enclosing wave
// processes a strided subset of the iter space: chunk `c` lands lane
// `lane * c + lin_tid`, an `scf.if lin < total` guards the trailing
// partial chunk, and the body runs once per in-range iteration with
// the unlinearized per-axis coords bound to the iter syms in scope.
// A closing `gpu.barrier` makes the cooperative writes visible to
// every thread before the per-lane readers downstream pick the
// finished tile back up.
//
// Body emission reuses the trivial-partition single-clone path: each
// in-range iteration loads one element per input operand, runs the
// body once with the loaded ins + a per-element outs init load, and
// stores the yielded scalars back at the same composed offset. The
// outs init load is the same race-free pattern the cooperative copy
// uses: each thread owns its element of the workgroup tile for the
// duration of the body, so the load+store pair never sees a write
// from another thread between them.
static LogicalResult lowerCollective(HCGenericOp op, ArrayRef<IterAxis> axes) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  auto tidAndSize = linearizedThreadAndSize(builder, loc, op);
  if (failed(tidAndSize))
    return op.emitOpError("collective dispatch requires a gpu.launch parent");
  auto [linTid, wgSize] = *tidAndSize;

  ValueRange bounds = op.getIterBounds();
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  Value total = c1;
  for (auto bound : bounds) {
    Value dim = castIdxToIndex(builder, loc, bound);
    total = arith::MulIOp::create(builder, loc, total, dim).getResult();
  }
  Value chunks =
      arith::CeilDivUIOp::create(builder, loc, total, wgSize).getResult();

  scf::ForOp loop =
      scf::ForOp::create(builder, loc, c0, chunks, c1, ValueRange{});
  LogicalResult bodyStatus = success();
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value chunkIdx = loop.getInductionVar();
    Value chunkOffset =
        arith::MulIOp::create(builder, loc, chunkIdx, wgSize).getResult();
    Value lin =
        arith::AddIOp::create(builder, loc, chunkOffset, linTid).getResult();
    Value inRange = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::ult, lin, total)
                        .getResult();
    auto rangeIf = scf::IfOp::create(builder, loc, TypeRange{}, inRange,
                                     /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard rangeGuard(builder);
      builder.setInsertionPointToStart(&rangeIf.getThenRegion().front());

      // Unlinearise `lin` into per-axis coords: rightmost axis varies
      // fastest, matching the row-major flatten convention used by
      // `hc-flatten-with-layouts` so post-flatten offset expressions
      // line up with the coord assignment.
      SmallVector<Value> coords(axes.size());
      Value remaining = lin;
      for (int axis = static_cast<int>(axes.size()) - 1; axis >= 0; --axis) {
        Value dim = castIdxToIndex(builder, loc, bounds[axis]);
        coords[axis] =
            arith::RemUIOp::create(builder, loc, remaining, dim).getResult();
        if (axis > 0)
          remaining =
              arith::DivUIOp::create(builder, loc, remaining, dim).getResult();
      }

      llvm::StringMap<Value> scope;
      seedAmbientScope(op, scope);
      for (auto [ax, coord] : llvm::zip(axes, coords))
        scope[ax.name] = coord;

      ArrayAttr insOff = op.getInsOffsetsAttr();
      SmallVector<Value> insVals(op.getIns().size());
      for (size_t ii = 0; ii < op.getIns().size(); ++ii) {
        ExprAttr origOff = getOperandOffset(insOff, ii);
        Value off = emitOffset(builder, loc, origOff, scope);
        Value ptr = op.getIns()[ii];
        Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
        Value addr =
            HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                .getResult();
        insVals[ii] =
            HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
      }

      ArrayAttr outsOff = op.getOutsOffsetsAttr();
      SmallVector<Value> outsVals(op.getOuts().size());
      for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
        ExprAttr origOff = getOperandOffset(outsOff, oi);
        Value off = emitOffset(builder, loc, origOff, scope);
        Value ptr = op.getOuts()[oi];
        Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
        Value addr =
            HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                .getResult();
        outsVals[oi] =
            HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
      }

      SmallVector<Value> yielded;
      if (failed(cloneBody(builder, op, insVals, outsVals, yielded))) {
        bodyStatus = failure();
      } else if (yielded.size() != op.getOuts().size()) {
        bodyStatus = op.emitOpError("body yielded wrong arity");
      } else {
        for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
          ExprAttr origOff = getOperandOffset(outsOff, oi);
          Value off = emitOffset(builder, loc, origOff, scope);
          Value ptr = op.getOuts()[oi];
          Value addr =
              HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                  .getResult();
          HCPtrStoreOp::create(builder, loc, yielded[oi], addr);
        }
      }
    }
  }
  if (failed(bodyStatus))
    return failure();

  // Make the cooperative writes visible to every thread of the
  // workgroup before any downstream per-lane read picks the
  // populated tile back up. Single-wave workgroups don't strictly
  // need it; the cost is trivial and the canonicalizer leaves it
  // alone deliberately.
  gpu::BarrierOp::create(builder, loc);
  return success();
}

// Pull the constant integer value off an iter bound. Caller checked
// `isV0Candidate` already, which guarantees one of the two forms
// resolves (`arith.constant index` or `!hc.idx<"<int>">`).
static int64_t constIterBound(Value bound) {
  if (auto def = bound.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(def.getValue()))
      return attr.getInt();
  if (auto idxTy = dyn_cast<IdxType>(bound.getType()))
    if (ExprAttr e = idxTy.getExpr())
      if (auto v = sym::getIntegerLiteralValue(e.getValue()))
        return *v;
  return -1;
}

// Per-axis row-major decompose of a flat lane index against a list of
// integer bounds. Rightmost axis varies fastest, matching the
// flatten convention used elsewhere. Returns `[i_0, i_1, ..., i_n-1]`
// for `lane in [0, prod(bounds))`.
static SmallVector<int, 4> decomposeRowMajor(int lane,
                                             ArrayRef<int64_t> bounds) {
  SmallVector<int, 4> vals(bounds.size(), 0);
  for (int axis = static_cast<int>(bounds.size()) - 1; axis >= 0; --axis) {
    int64_t b = bounds[axis];
    vals[axis] = lane % static_cast<int>(b);
    lane = lane / static_cast<int>(b);
  }
  return vals;
}

// Builder helper for the scope binding `iter_sym -> arith.constant 0`
// used by the value-outs path: `substituteIterDeltas` shifts each
// iter sym by its per-lane delta, so binding the residual sym to `0`
// makes `emitOffset` list the iter sym as an explicit operand and
// keeps the launch-body apply lowering's resolve invariant happy
// — same shape as the partition path's `scope[sym] = induction_var`
// binding, just with a constant for the residual.
static llvm::StringMap<Value>
buildZeroIterScope(OpBuilder &builder, Location loc, ArrayRef<IterAxis> axes) {
  Value cZero = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  llvm::StringMap<Value> scope;
  for (const IterAxis &ax : axes)
    scope[ax.name] = cZero;
  return scope;
}

// Value-typed outs lowering: fully unroll the parallel iter space
// at compile time (no `scf.parallel` — the result has to live in
// one SSA register so the sweep IS the unroll), reuse the per-lane
// ins-load + body-clone infrastructure from the partition path, and
// compose the per-parLane scalar finals into a single
// `vector.from_elements` per value-typed out. Ptr-typed outs ride
// the same per-parLane emission with contig-group merging on the
// stores; mixed outs (some ptr, some value) work since the loop
// shape is per-lane either way.
//
// Slot mapping: the outs offset expression must evaluate to a
// bijection on `[0, prodPar)` when iter syms substitute to their
// row-major delta values — the verifier in `isV0Candidate` already
// pinned the offset's free syms to iter syms only, so the
// `substituteIterValues` step produces a pure integer for each lane.
//
// `lowerOne` dispatches here when any out is value-typed; the
// partition path handles all-ptr-out generics where the
// `scf.parallel`-shaped emission is the right form.
static LogicalResult lowerValueOuts(HCGenericOp op, ArrayRef<IterAxis> axes) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

  SmallVector<int64_t> bounds;
  bounds.reserve(axes.size());
  for (const IterAxis &ax : axes) {
    int64_t b = constIterBound(ax.bound);
    if (b < 0)
      return op.emitOpError(
          "value-outs lowering requires constant iter bounds");
    bounds.push_back(b);
  }
  int64_t prodPar = 1;
  for (int64_t b : bounds)
    prodPar *= b;

  // Verify every value-typed out has a rank-1 integer shape of the
  // same width — one slot per parLane. Ptr outs skip the check.
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType()))
      continue;
    auto count = getValueOutLaneCount(v.getType());
    if (!count || *count != prodPar)
      return op.emitOpError("value-outs lowering: outs #")
             << oi << " has lane count " << (count ? *count : -1)
             << " but parallel iter space has " << prodPar << " lanes";
  }

  // Per-parLane slot for each value-typed out via constant-eval of
  // the outs offset. Bijection check rejects gaps / collisions; the
  // diagnostic-guard bead can tighten this further with a more
  // helpful message once the surface stabilizes.
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  size_t numOuts = op.getOuts().size();
  size_t numIns = op.getIns().size();
  SmallVector<StringRef> iterNames;
  iterNames.reserve(axes.size());
  for (const IterAxis &ax : axes)
    iterNames.push_back(ax.name);
  SmallVector<SmallVector<int64_t>> slotPerOut(numOuts);
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType()))
      continue;
    ExprAttr origOff = getOperandOffset(outsOff, oi);
    SmallVector<int64_t> slots;
    slots.reserve(prodPar);
    llvm::SmallDenseSet<int64_t, 16> seen;
    for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
      SmallVector<int, 4> vals =
          decomposeRowMajor(static_cast<int>(parLane), bounds);
      sym::ExprHandle subbed =
          substituteIterValues(store, origOff.getValue(), iterNames, vals);
      std::optional<int64_t> slot = sym::getIntegerLiteralValue(subbed);
      if (!slot || *slot < 0 || *slot >= prodPar || !seen.insert(*slot).second)
        return op.emitOpError("value-outs lowering: outs #")
               << oi << " offset is not a bijection on [0, " << prodPar
               << ") at parLane " << parLane;
      slots.push_back(*slot);
    }
    slotPerOut[oi] = std::move(slots);
  }

  // Reuse the partition emission machinery with `p[a] = bounds[a]`
  // on every axis (full unroll) and declaration order. The ins-load
  // path then runs `findContigGroups` over the unrolled lane offsets
  // and merges contiguous runs into vector loads (a 16x16 row-major
  // load with unit-stride inner axis collapses to one
  // `vector<256xT>` load before LLVM's vectorizer ever sees the IR).
  AxisOrder order;
  order.reserve(axes.size());
  Partition p;
  p.reserve(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    order.push_back(i);
    p.push_back(static_cast<int>(bounds[i]));
  }

  llvm::StringMap<Value> scope = buildZeroIterScope(builder, loc, axes);
  seedAmbientScope(op, scope);
  SmallVector<SmallVector<Value>> insLanes =
      emitInsLoadsLaned(builder, op, axes, order, p, scope, store);

  // Per-parLane outs init: value-typed ones extract from the SSA out
  // (via UCC if the carrier isn't builtin `vector`); ptr-typed ones
  // ride the same per-lane init-load path the partition emitter uses,
  // with contig-group merging on the load side.
  SmallVector<bool, 4> parMask(axes.size(), true);
  SmallVector<SmallVector<Value>> outsInit(prodPar,
                                           SmallVector<Value>(numOuts));
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (auto pt = dyn_cast<PtrType>(v.getType())) {
      Type elemTy = pt.getElementType();
      ExprAttr origOff = getOperandOffset(outsOff, oi);
      SmallVector<sym::ExprHandle> offs =
          laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
      auto groups = findContigGroups(store, offs);
      SmallVector<Value> initSlots(prodPar);
      for (const ContigGroup &g : groups) {
        Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes,
                                   order, p, parMask, g.start);
        Value addr = HCPtrOffsetOp::create(builder, loc, v.getType(), v, off)
                         .getResult();
        emitGroupLoad(builder, loc, v, elemTy, addr, g, initSlots);
      }
      for (int64_t parLane = 0; parLane < prodPar; ++parLane)
        outsInit[parLane][oi] = initSlots[parLane];
      continue;
    }
    // Value-typed out: per-parLane init = `vector.extract` at slot.
    // The slot is the integer the offset evaluates to for that lane;
    // it's not necessarily `parLane` (the offset might permute).
    auto count = getValueOutLaneCount(v.getType());
    auto shaped = cast<SymbolicallyShapedTypeInterface>(v.getType());
    Type elemTy = shaped.getSymbolicElementType();
    mlir::VectorType vecTy = mlir::VectorType::get({*count}, elemTy);
    Value asVec = v;
    if (v.getType() != Type(vecTy))
      asVec = UnrealizedConversionCastOp::create(builder, loc, vecTy, v)
                  .getResult(0);
    for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
      int64_t slot = slotPerOut[oi][parLane];
      outsInit[parLane][oi] = vector::ExtractOp::create(builder, loc, asVec,
                                                        ArrayRef<int64_t>{slot})
                                  .getResult();
    }
  }

  // Per-parLane body clones. Iter syms have no SSA representation in
  // the body (they're symbolic in offset exprs only), so the clone
  // mapping just covers ins + outs block args.
  SmallVector<SmallVector<Value>> finals(prodPar, SmallVector<Value>(numOuts));
  for (int64_t lane = 0; lane < prodPar; ++lane) {
    SmallVector<Value> insVals(numIns);
    for (size_t ii = 0; ii < numIns; ++ii)
      insVals[ii] = insLanes[ii][lane];
    SmallVector<Value> yielded;
    if (failed(cloneBody(builder, op, insVals, outsInit[lane], yielded)))
      return failure();
    if (yielded.size() != numOuts)
      return op.emitOpError("body yielded wrong arity");
    finals[lane] = std::move(yielded);
  }

  // Ptr-typed outs ride the same per-parLane store path the partition
  // emitter uses, with contig-group merging on the writes. Value-
  // typed outs compose all finals through one `vector.from_elements`
  // sized to the result vector — the slot mapping placed each
  // parLane's final at its compile-time slot index.
  SmallVector<Value, 2> opResults(op.getNumResults());
  size_t resultIdx = 0;
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (auto pt = dyn_cast<PtrType>(v.getType())) {
      Type elemTy = pt.getElementType();
      ExprAttr origOff = getOperandOffset(outsOff, oi);
      SmallVector<sym::ExprHandle> offs =
          laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
      auto groups = findContigGroups(store, offs);
      SmallVector<Value> laneFinals(prodPar);
      for (int64_t parLane = 0; parLane < prodPar; ++parLane)
        laneFinals[parLane] = finals[parLane][oi];
      for (const ContigGroup &g : groups) {
        Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes,
                                   order, p, parMask, g.start);
        Value addr = HCPtrOffsetOp::create(builder, loc, v.getType(), v, off)
                         .getResult();
        emitGroupStore(builder, loc, elemTy, addr, g, laneFinals);
      }
      continue;
    }
    auto count = getValueOutLaneCount(v.getType());
    auto shaped = cast<SymbolicallyShapedTypeInterface>(v.getType());
    Type elemTy = shaped.getSymbolicElementType();
    mlir::VectorType vecTy = mlir::VectorType::get({*count}, elemTy);
    SmallVector<Value> elems(*count);
    for (int64_t parLane = 0; parLane < prodPar; ++parLane)
      elems[slotPerOut[oi][parLane]] = finals[parLane][oi];
    Value vec =
        vector::FromElementsOp::create(builder, loc, vecTy, elems).getResult();
    Value asOrig = vec;
    if (v.getType() != Type(vecTy))
      asOrig =
          UnrealizedConversionCastOp::create(builder, loc, v.getType(), vec)
              .getResult(0);
    opResults[resultIdx++] = asOrig;
  }
  if (resultIdx != op.getNumResults())
    return op.emitOpError("value-outs lowering: result/out arity mismatch");

  op->replaceAllUsesWith(opResults);
  return success();
}

// Top-level: pick `(order, partition)` via the divisibility-pruned
// merge-score search, then dispatch to the partition-aware emitter.
// The trivial `(1, ..., 1)` partition collapses through the same
// path verb-for-verb to the scalar baseline (single scalar load /
// store per operand, body cloned once per innermost iteration).
//
// Collective candidates (workgroup-shared outs inside a launch)
// route through `lowerCollective` instead — the chunk-and-publish
// shape is the right loop nest for cooperative tile population, and
// running `lowerWithPartition` against shared outs would have every
// thread of the wave fight every other thread for every element.
//
// Value-typed outs route through `lowerValueOuts` regardless of
// the launch / workgroup-out signal — the result has to live in a
// single SSA register, so the parallel sweep must be compile-time-
// unrolled rather than scattered across `scf.parallel` iterations.
static LogicalResult lowerOne(HCGenericOp op) {
  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallVector<IterAxis> axes = collectIterAxes(op);

  if (hasValueOuts(op)) {
    if (failed(lowerValueOuts(op, axes)))
      return failure();
    op.erase();
    return success();
  }

  if (isCollectiveCandidate(op, axes)) {
    if (failed(lowerCollective(op, axes)))
      return failure();
  } else {
    auto [order, p] = selectBest(op, axes, store);
    if (failed(lowerWithPartition(op, axes, order, p)))
      return failure();
  }

  // Ptr-only outs produce zero SSA results, so erasing is sufficient.
  // The candidate check + `hasValueOuts` early-return upstream
  // guarantees this branch only sees zero-result generics; the
  // belt-and-suspenders error keeps a future relaxation of either
  // gate from silently losing results.
  if (op.getNumResults() != 0)
    return op.emitOpError("ptr-out lowering only handles all-ptr outs "
                          "(zero SSA results)");
  op.erase();
  return success();
}

struct HCLowerGenericPass
    : public hc::impl::HCLowerGenericBase<HCLowerGenericPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<HCGenericOp> work;
    getOperation()->walk([&](HCGenericOp op) { work.push_back(op); });
    for (HCGenericOp op : work) {
      if (!isV0Candidate(op))
        continue;
      if (failed(lowerOne(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

// `createHCLowerGenericPass()` is emitted by tablegen.
