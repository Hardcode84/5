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
#include "llvm/Support/raw_ostream.h"

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

// True when this op has at least one value-typed (non-ptr) ins
// operand. The fully-unrolled emitter materializes those via per-lane
// `vector.extract` against a compile-time slot index (gather pattern);
// the partition-aware emitter has no shape for that — it loads each
// lane through `hc.ptr_load`, which requires a `!hc.ptr` carrier. So
// value-typed ins force the same fully-unrolled dispatch as value-
// typed outs.
static bool hasValueIns(HCGenericOp op) {
  for (Value v : op.getIns())
    if (!isa<PtrType>(v.getType()))
      return true;
  return false;
}

// Pre-flight gate: returns `nullopt` when this op fits one of the
// three lowering paths in `lowerOne` (collective dispatch, value-
// outs unroll, scalar partition), otherwise a short reason naming
// the specific check that failed. Callers use the boolean coercion
// of the return (`has_value()` ↔ rejected) for the dispatch
// decision and the underlying string for the diagnostic at the end
// of the pass.
//
// The ptr-only path accepts any rank-1 offset with resolved iter
// bounds. The value-typed path additionally requires constant iter
// bounds (full unroll happens at compile time), all iters parallel
// (a reduction iter would need cross-lane accumulator carry the
// single-SSA-result boundary doesn't model), a plain `hc.yield` /
// `hc.yield_predicated` terminator, and every value-side offset's
// free syms confined to iter syms — anything else makes the slot
// index ambient-dependent and the compile-time slot evaluation
// can't pin it. Opaque pointers (no element type on the carrier)
// also bail: load emission needs the element type to spell the
// result and the pointer's `$elementType` is the only available
// source.
//
// We deliberately diagnose every rejection at the end of the pass
// instead of silently skipping: an un-lowered `hc.generic` flowing
// through GPU outlining + ROCDL attach trips downstream passes with
// cryptic errors far from the source, and the supported scope here
// is narrow enough that "this op didn't lower" is always a real bug
// at the frontend or in an earlier rewrite, never an intentional
// fallback.
static std::optional<std::string> diagnoseUnsupported(HCGenericOp op) {
  auto fmt = [](auto &&...args) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    (os << ... << args);
    return buf;
  };
  auto typeStr = [](Type t) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    t.print(os);
    return buf;
  };
  auto goodPtr = [](Value v) {
    auto p = dyn_cast<PtrType>(v.getType());
    return p && p.getElementType();
  };
  // Value-typed ins are accepted when the carrier resolves to a
  // compile-time lane count (rank-1, integer-literal shape — same
  // gate the outs side uses). The actual per-lane slot evaluation
  // happens in `lowerValueOuts`; the candidate gate just admits the
  // op into the fully-unrolled path.
  bool hasValIn = false;
  for (auto [i, v] : llvm::enumerate(op.getIns())) {
    if (goodPtr(v))
      continue;
    if (!getValueOutLaneCount(v.getType()))
      return fmt("ins #", i, " has type '", typeStr(v.getType()),
                 "'; expected !hc.ptr<...> with element type or a rank-1 "
                 "fixed-lane carrier with integer-literal shape");
    hasValIn = true;
  }
  bool hasValOut = false;
  for (auto [i, v] : llvm::enumerate(op.getOuts())) {
    if (goodPtr(v))
      continue;
    if (!getValueOutLaneCount(v.getType()))
      return fmt("outs #", i, " has type '", typeStr(v.getType()),
                 "'; expected !hc.ptr<...> with element type or a rank-1 "
                 "fixed-lane carrier with integer-literal shape");
    hasValOut = true;
  }
  // No rank-N offset check here — the op verifier already pins
  // per-operand offset arity to the operand's rank (ptr → 1, rank-N
  // shaped → N), and the operand-type gate above rejects every
  // rank-N shaped type that would let a rank-N offset slip through.
  // The combination "rank-1 operand, rank-N offsets" is not
  // representable in textual IR. A future verifier relaxation should
  // restore the check (and pair it with a LIT fixture that actually
  // exercises it).
  ArrayAttr insOff = op.getInsOffsetsAttr();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (auto [i, bound] : llvm::enumerate(op.getIterBounds()))
    if (auto def = bound.getDefiningOp())
      if (isa<HCUndefValueOp>(def))
        return fmt("iter #", i,
                   " bound is hc.undef_value; bounds must resolve to a "
                   "concrete index value");
  for (Operation &nested : op.getBody().front())
    if (isa<scf::IfOp>(nested))
      return std::string(
          "body contains nested scf.if; control flow inside the body is not "
          "modeled (lower predicates to hc.yield_predicated first)");
  if (!hasValOut && !hasValIn)
    return std::nullopt;

  // Fully-unrolled-path conditions (any value-typed operand). Iter
  // bounds: each one must resolve to a compile-time integer (either
  // an `arith.constant` index value or an `!hc.idx<"<int>">`-typed
  // SSA bound — the frontend planting `hc.const` against an integer-
  // only shape is the canonical case).
  llvm::StringSet<> iterNames;
  for (Attribute s : op.getIterSymsAttr())
    iterNames.insert(cast<StringAttr>(s).getValue());
  for (auto [i, bound] : llvm::enumerate(op.getIterBounds())) {
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
      return fmt("iter #", i,
                 " bound is not a compile-time non-negative integer literal "
                 "(value-typed operand needs constant bound)");
  }
  // All iters must be parallel — the unrolled emitter threads every
  // parLane through the result vector / store sequence, and a
  // reduction iter would need cross-lane carry that the boundary
  // form doesn't model. Value-typed ins reuse the same constraint:
  // the gather slot is a function of iter syms alone, and a reduction
  // iter would mean the same value-in lane gets read at different
  // reduction steps with no scf-loop carry to express it.
  for (auto [i, k] : llvm::enumerate(op.getIterKindsAttr()))
    if (cast<IterKindAttr>(k).getValue() != IterKind::Parallel)
      return fmt("iter #", i,
                 " kind is reduction (value-typed operand needs all-parallel "
                 "iters)");
  // Body terminator: `hc.yield` (unconditional publish) or
  // `hc.yield_predicated` (per-value mask gate → `arith.select` at
  // the boundary in `cloneBody`). Anything else escaped the emitters
  // we know about.
  Operation &term = op.getBody().front().back();
  if (!isa<HCYieldOp, HCYieldPredicatedOp>(&term))
    return fmt("body terminator '", term.getName().getStringRef(),
               "' is not hc.yield or hc.yield_predicated");
  // Outs offset's free syms must be a subset of iter syms; ambient
  // syms (shape / stride params) would make slot evaluation
  // ambient-dependent. Value-typed ins offsets are the per-lane slot
  // expressions and follow the same constraint — `lowerValueOuts`
  // constant-evaluates them at every parallel-lane combo.
  auto findAmbientSym = [&](ArrayAttr arr, OperandRange operands,
                            StringRef kind) -> std::optional<std::string> {
    for (size_t i = 0, e = arr.size(); i < e; ++i) {
      Value v = operands[i];
      // Ptr operands carry their offset against the source pointer
      // and may reference ambient syms (the `emitOffset` path
      // resolves them via `loopScope` + ambient bindings); only the
      // value-typed operands need slot-eval, and slot-eval needs
      // iter-only.
      if (isa<PtrType>(v.getType()))
        continue;
      auto off = cast<ExprAttr>(cast<ArrayAttr>(arr[i])[0]);
      std::optional<std::string> ambient;
      sym::walkSymbolNames(off.getValue(), [&](StringRef name) {
        if (!iterNames.contains(name) && !ambient)
          ambient = name.str();
      });
      if (ambient)
        return fmt(kind, " #", i, " offset references non-iter symbol '",
                   *ambient, "' (value-typed operand needs iter-only offsets)");
    }
    return std::nullopt;
  };
  if (auto r = findAmbientSym(insOff, op.getIns(), "ins"))
    return r;
  if (auto r = findAmbientSym(outsOff, op.getOuts(), "outs"))
    return r;
  return std::nullopt;
}

// Pull the single composed offset expression for an operand at
// position `idx` from an `ins_offsets` / `outs_offsets` array.
// Caller has already checked rank-1 via `diagnoseUnsupported`.
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

// Body-authoring convention for `hc.generic`: an op inside the body
// may reference iter syms by name in its `!hc.idx<...>` / `!hc.pred<...>`
// expression with no explicit binding to a runtime value — the
// expectation is the lowering supplies the binding when it realises a
// specific lane. The mask emitters (`hc-load-store-to-generic`'s
// `hc.load_mask` case, future OOB-tile guards) lean on this so the
// body can compute `(lo + step * i_0) < D0` without first knowing
// what concrete value `i_0` has on this iteration.
//
// `cloned` was just produced by `builder.clone(&nested, mapping)`. If
// it's an `hc.idx_apply` / `hc.pred_apply` whose expression references
// iter syms in `iterScope` but doesn't already list them as bindings,
// replace it with an augmented op that pins each missing iter sym to
// `iterScope[sym]`. The expression is preserved verbatim — bindings
// only provide runtime integers for free symbols, they don't rewrite
// the symbolic form. Updates `mapping` so subsequent body clones that
// reference `nested`'s results see the augmented op's; replaces the
// stale clone's uses and erases it.
//
// No-op for body ops that aren't `hc.idx_apply` / `hc.pred_apply`,
// for empty `iterScope` (the partition / reduction call sites today),
// and for ops whose expression doesn't reference any iter sym in
// scope. The path is cheap on the common case — one lookup per body
// op, one walk over its expression's free names.
static void bindIterSymsInClone(OpBuilder &builder, Operation &nested,
                                Operation *cloned,
                                const llvm::StringMap<Value> &iterScope,
                                IRMapping &mapping) {
  if (iterScope.empty())
    return;
  ArrayAttr existingSyms;
  bool isIdx = false;
  if (auto idx = dyn_cast<HCIdxApplyOp>(cloned)) {
    auto idxTy = dyn_cast<IdxType>(idx.getResult().getType());
    if (!idxTy || !idxTy.getExpr())
      return;
    existingSyms = idx.getSymbolsAttr();
    isIdx = true;
  } else if (auto pred = dyn_cast<HCPredApplyOp>(cloned)) {
    auto predTy = dyn_cast<PredType>(pred.getResult().getType());
    if (!predTy || !predTy.getPred())
      return;
    existingSyms = pred.getSymbolsAttr();
  } else {
    return;
  }

  llvm::StringSet<> already;
  for (Attribute n : existingSyms)
    already.insert(cast<StringAttr>(n).getValue());

  SmallVector<StringRef, 4> additions;
  auto walker = [&](StringRef name) {
    if (already.contains(name))
      return;
    if (!iterScope.count(name))
      return;
    if (llvm::is_contained(additions, name))
      return;
    additions.push_back(name);
  };
  if (isIdx) {
    auto idxTy = cast<IdxType>(cloned->getResult(0).getType());
    sym::walkSymbolNames(idxTy.getExpr().getValue(), walker);
  } else {
    auto predTy = cast<PredType>(cloned->getResult(0).getType());
    sym::walkSymbolNames(predTy.getPred().getValue(), walker);
  }
  if (additions.empty())
    return;

  // Sort for deterministic textual form across rebuilds — see the
  // matching note on `emitOffset`.
  llvm::sort(additions);

  SmallVector<Value> newOperands(cloned->getOperands());
  SmallVector<Attribute> newSyms(existingSyms.begin(), existingSyms.end());
  for (StringRef name : additions) {
    newOperands.push_back(iterScope.lookup(name));
    newSyms.push_back(builder.getStringAttr(name));
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(cloned);
  Operation *replacement;
  if (isIdx) {
    replacement = HCIdxApplyOp::create(
        builder, cloned->getLoc(), cloned->getResult(0).getType(), newOperands,
        builder.getArrayAttr(newSyms));
  } else {
    replacement = HCPredApplyOp::create(
        builder, cloned->getLoc(), cloned->getResult(0).getType(), newOperands,
        builder.getArrayAttr(newSyms));
  }
  for (auto [origRes, newRes] :
       llvm::zip_equal(nested.getResults(), replacement->getResults()))
    mapping.map(origRes, newRes);
  cloned->replaceAllUsesWith(replacement);
  cloned->erase();
}

// UCC a per-lane mask to the type `arith.select` requires (`i1` for
// scalar selects, `vector<Nxi1>` for vector selects). Inputs at this
// boundary are whatever the body's mask block arg resolved to —
// commonly `!hc.pred` (which the broader pipeline lowers to `i1`
// downstream) or already `i1`. The UCC pair canonicalises away when
// the producer side is the matching i1.
static Value coerceMaskToI1(OpBuilder &builder, Location loc, Value mask,
                            Type valueType) {
  Type want;
  if (auto vt = dyn_cast<mlir::VectorType>(valueType))
    want = mlir::VectorType::get(vt.getShape(), builder.getI1Type());
  else
    want = builder.getI1Type();
  if (mask.getType() == want)
    return mask;
  return UnrealizedConversionCastOp::create(builder, loc, want, mask)
      .getResult(0);
}

// Body cloner. Replaces block-arg references with their loaded /
// carry values via the IR mapping, lowers the body's terminator, and
// emits the cloned ops at the builder's current insertion point.
//
// Terminator handling:
//   * `hc.yield` — surface the yielded SSAs verbatim through
//     `yieldedOut` for the caller to feed into the enclosing
//     `scf.for` carry or the per-lane store path.
//   * `hc.yield_predicated` — position `i` becomes
//     `arith.select(mask, val, outsVals[i])`, so masked-out lanes
//     preserve the init (the outs-as-init carry the per-lane
//     init-load just published). Mask is coerced to `i1` /
//     `vector<Nxi1>` via UCC because the body-side carrier is often
//     `!hc.pred` and the broader pipeline owns the actual
//     pred-to-i1 resolution.
static LogicalResult cloneBody(OpBuilder &builder, HCGenericOp op,
                               ValueRange insVals, ValueRange outsVals,
                               const llvm::StringMap<Value> &iterScope,
                               SmallVectorImpl<Value> &yieldedOut) {
  Block &src = op.getBody().front();
  IRMapping mapping;
  size_t pos = 0;
  for (Value v : insVals)
    mapping.map(src.getArgument(pos++), v);
  for (Value v : outsVals)
    mapping.map(src.getArgument(pos++), v);
  Operation &term = src.back();
  for (Operation &nested : src) {
    if (&nested == &term)
      break;
    Operation *cloned = builder.clone(nested, mapping);
    bindIterSymsInClone(builder, nested, cloned, iterScope, mapping);
  }
  yieldedOut.clear();
  if (auto yield = dyn_cast<HCYieldOp>(&term)) {
    yieldedOut.reserve(yield.getValues().size());
    for (Value v : yield.getValues())
      yieldedOut.push_back(mapping.lookupOrDefault(v));
    return success();
  }
  if (auto pyield = dyn_cast<HCYieldPredicatedOp>(&term)) {
    auto vals = pyield.getValues();
    auto masks = pyield.getMasks();
    if (vals.size() != outsVals.size())
      return op.emitOpError(
          "yield_predicated arity does not match outs operand count");
    if (vals.size() != masks.size())
      return op.emitOpError("yield_predicated values/masks size mismatch");
    yieldedOut.reserve(vals.size());
    Location loc = op.getLoc();
    for (auto [v, m, init] : llvm::zip_equal(vals, masks, outsVals)) {
      Value mapped = mapping.lookupOrDefault(v);
      Value mappedMask = mapping.lookupOrDefault(m);
      Value i1Mask = coerceMaskToI1(builder, loc, mappedMask, mapped.getType());
      Value sel = arith::SelectOp::create(builder, loc, i1Mask, mapped, init)
                      .getResult();
      yieldedOut.push_back(sel);
    }
    return success();
  }
  return op.emitOpError(
      "body must end with `hc.yield` or `hc.yield_predicated`");
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

// Emit per-lane loads for every ptr-typed input of `op`, returning a
// flat `[in_idx][lane]` 2D buffer of SSA values. Per-input contig
// groups drive load shape: scalar groups emit single `hc.ptr_load`,
// groups of size > 1 emit vector loads + extracts. Lanes are
// decomposed against every iter axis (mask = all-true). Value-typed
// ins are left empty in `result` — only the fully-unrolled emitter
// reaches them, and it fills the entries via `vector.extract` against
// the precomputed gather-slot table; the partition path never sees
// value-typed ins (`diagnoseUnsupported` gates them through the unrolled
// path) so leaving the entry empty there is unreachable.
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
    Value ptr = op.getIns()[ii];
    if (!isa<PtrType>(ptr.getType()))
      continue;
    ExprAttr origOff = getOperandOffset(insOff, ii);
    SmallVector<sym::ExprHandle> offs =
        laneOffsets(store, origOff, axes, order, p, allMask, prodAll);
    auto groups = findContigGroups(store, offs);
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
    // Partition / reduction path doesn't expose iter-sym SSA bindings
    // to the body yet — emitInsLoadsLaned bakes deltas into the
    // operand-offset side, and the body's only iter-sym story today is
    // the value-outs unroll in `lowerValueOuts`. Pass an empty scope so
    // `bindIterSymsInClone` no-ops here. Wiring this up properly is a
    // follow-up if a body op ever references an iter sym from the
    // partition path; the lane induction var is in `scope`, so the
    // augmentation would emit one `+ delta_a` constant per axis.
    llvm::StringMap<Value> emptyIterScope;
    if (failed(
            cloneBody(builder, op, insVals, outsVals, emptyIterScope, yielded)))
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

// Element type for the `!hc.ptr<workgroup, T>` a `bare_tensor` outs
// resolves through. Matches `convertElementType` in
// `HCLowerLaunchBodyPass.cpp` so the UCC we plant against the outs
// type-matches the one `ConvertNullaryShapedConstantOp` already
// planted on the producing side (`hc.zeros : bare_tensor` →
// `hc.alloc workgroup` + UCC back). Identical converted-element
// types are what lets canonicalize fold the
// `ptr<workgroup> → bare_tensor → ptr<workgroup>` pair to the
// underlying alloc.
static Type convertBareTensorElement(Type t) {
  if (isa<PredType>(t))
    return IntegerType::get(t.getContext(), 1);
  if (t.isIntOrIndexOrFloat())
    return t;
  return {};
}

// `!hc.ptr<workgroup, T>` the converter pins for a `bare_tensor` outs.
// `BareTensorType` itself doesn't carry a ptr, but every bare-tensor
// SSA inside `gpu.launch` traces to an `hc.alloc workgroup` through
// the source-materialization UCC the partial-conversion driver
// planted in `hc-lower-launch-body`. We rebuild the same ptr type
// here so a fresh `bare_tensor → ptr<workgroup>` UCC pairs with the
// upstream one and folds out at canonicalize.
static PtrType workgroupPtrFor(BareTensorType bt) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(bt);
  Type elem = convertBareTensorElement(shaped.getSymbolicElementType());
  if (!elem)
    return PtrType();
  return PtrType::get(bt.getContext(), AddrSpace::Workgroup, elem);
}

// Collective dispatch detector: at least one outs operand is
// `!hc.ptr<workgroup, T>` (workgroup-staged tile) or a `bare_tensor` view
// (every bare-tensor SSA inside `gpu.launch` is workgroup-backed by
// the launch-body type-converter convention — `hc.zeros : bare_tensor`
// collapsed to `hc.alloc workgroup` + a UCC bridge to the still-
// bare_tensor consumer slot). Every iter is parallel (no cross-
// thread accumulation) and the op sits inside a `gpu.launch` (we
// need the dim3 thread/block layout to chunk the iter space across
// the wave). Per-lane outs falls through to the existing partition-
// aware path — running scf.parallel without partitioning is the
// correct shape for lane-local results.
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
    if (auto ptr = dyn_cast<PtrType>(v.getType())) {
      if (ptr.getAddrSpace() == AddrSpace::Workgroup)
        hasWorkgroupOut = true;
      continue;
    }
    if (auto bt = dyn_cast<BareTensorType>(v.getType())) {
      if (!workgroupPtrFor(bt))
        return false;
      hasWorkgroupOut = true;
      continue;
    }
    return false;
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
// stores the yielded scalars back at the same composed offset. Each
// thread owns its element of the workgroup tile for the duration of
// the body, so the load+store pair never sees a write from another
// thread between them; the closing `gpu.barrier` makes the chunk's
// writes visible before the per-lane readers run.
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

  // Pin every outs to a workgroup ptr we can ptr_offset/load/store
  // against. Direct ptr outs are themselves; bare-tensor outs UCC
  // through to their backing ptr<workgroup> type once, hoisted above
  // the chunk loop so the cast doesn't re-emit per chunk. The
  // upstream UCC `ptr<workgroup> → bare_tensor` (planted by
  // `hc-lower-launch-body`'s shaped-constant lowering on the
  // producing side) and this fresh `bare_tensor → ptr<workgroup>`
  // form a foldable pair: canonicalize collapses the chain back to
  // the original `hc.alloc workgroup` so per-thread stores hit the
  // real LDS storage without a UCC dead-end at LLVM translation.
  SmallVector<Value> outsPtrs(op.getOuts().size());
  for (auto [i, out] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(out.getType())) {
      outsPtrs[i] = out;
      continue;
    }
    auto bt = cast<BareTensorType>(out.getType());
    PtrType ptrTy = workgroupPtrFor(bt);
    assert(ptrTy && "isCollectiveCandidate accepted unconvertible bare_tensor");
    outsPtrs[i] = UnrealizedConversionCastOp::create(builder, loc, ptrTy, out)
                      .getResult(0);
  }

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
      // fastest, matching the flatten convention used by
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
        Value loaded =
            HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
        // `bare_tensor` ins with a non-builtin element type (e.g.
        // `!hc.pred`) ride through workgroup LDS as their builtin
        // surrogate (`i1`). The body's block arg keeps the symbolic
        // type, so bridge back through a UCC before substitution —
        // mirrors the value-ins lane in `lowerValueOuts`.
        Type bodyArgTy = op.getBody().front().getArgument(ii).getType();
        if (loaded.getType() != bodyArgTy)
          loaded = UnrealizedConversionCastOp::create(builder, loc, bodyArgTy,
                                                      loaded)
                       .getResult(0);
        insVals[ii] = loaded;
      }

      // `outsPtrs` was set up above the chunk loop (direct ptr outs
      // verbatim; bare-tensor outs UCC'd through once to the
      // converter-pinned `ptr<workgroup>`). Access emission below
      // walks it for every operand uniformly.
      ArrayAttr outsOff = op.getOutsOffsetsAttr();
      SmallVector<Value> outsVals(op.getOuts().size());
      for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
        ExprAttr origOff = getOperandOffset(outsOff, oi);
        Value off = emitOffset(builder, loc, origOff, scope);
        Value ptr = outsPtrs[oi];
        Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
        Value addr =
            HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                .getResult();
        Value loaded =
            HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
        // Same coercion as the ins side: outs block args are the
        // operand's symbolic element type, the LDS store is the
        // builtin surrogate. UCC bridges the gap.
        Type bodyArgTy =
            op.getBody().front().getArgument(op.getIns().size() + oi).getType();
        if (loaded.getType() != bodyArgTy)
          loaded = UnrealizedConversionCastOp::create(builder, loc, bodyArgTy,
                                                      loaded)
                       .getResult(0);
        outsVals[oi] = loaded;
      }

      SmallVector<Value> yielded;
      // Hand the chunk's iter-sym SSA bindings to `cloneBody` so any
      // body `hc.pred_apply` / `hc.idx_apply` that references the loop's
      // iter syms (e.g. the `hc.load_mask` rewrite plants a single
      // pred_apply with the conjunction predicate referring to `i_0` /
      // `i_1` as free symbols) gets explicit-operand bindings before
      // the second `hc-lower-launch-body` invocation lowers it. The
      // `scope` map already mixes ambient and iter sym SSA per the
      // offset-emission setup above; `bindIterSymsInClone` only acts on
      // names the apply's pred/expr actually references and that aren't
      // already in the apply's `symbols` list, so any unused entries
      // are harmless.
      if (failed(cloneBody(builder, op, insVals, outsVals, scope, yielded))) {
        bodyStatus = failure();
      } else if (yielded.size() != op.getOuts().size()) {
        bodyStatus = op.emitOpError("body yielded wrong arity");
      } else {
        for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
          ExprAttr origOff = getOperandOffset(outsOff, oi);
          Value off = emitOffset(builder, loc, origOff, scope);
          Value ptr = outsPtrs[oi];
          Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
          Value addr =
              HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                  .getResult();
          // Yielded value is in the body's element type (e.g.
          // `!hc.pred`); the LDS pointer is the builtin surrogate
          // (e.g. `i1`). UCC at the boundary keeps `hc.ptr_store`'s
          // verifier happy.
          Value toStore = yielded[oi];
          if (toStore.getType() != elemTy)
            toStore = UnrealizedConversionCastOp::create(builder, loc, elemTy,
                                                         toStore)
                          .getResult(0);
          HCPtrStoreOp::create(builder, loc, toStore, addr);
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

  // The generic's result (`bare_tensor` value, when present) is the
  // post-write logical view of the same storage the cooperative
  // emit just populated. Replace it with the original outs SSA so
  // downstream consumers (typically a follow-up UCC back to
  // `!hc.ptr<workgroup>`) see the input chain and fold cleanly,
  // instead of dangling against the erased generic.
  for (auto [res, out] : llvm::zip(op.getResults(), op.getOuts()))
    res.replaceAllUsesWith(out);
  return success();
}

// Pull the constant integer value off an iter bound. Caller checked
// `diagnoseUnsupported` already, which guarantees one of the two forms
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

// Per-axis decompose of a flat lane index against a list of integer
// bounds. Rightmost axis varies fastest, matching the flatten
// convention used elsewhere. Returns `[i_0, i_1, ..., i_n-1]` for
// `lane in [0, prod(bounds))`.
static SmallVector<int, 4> decomposeLaneIndex(int lane,
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

// Compile-time slot for a value-typed operand: returns the element
// type if it can be wrapped in a builtin `VectorType` (floats, ints,
// index, or `!hc.pred` → `i1`). Returns null for everything else.
// The UCC chain through `vector<NxBuiltin>` is the only mechanism
// the per-lane extract path can use, so non-wrappable element types
// have no path through this emitter.
static Type asBuiltinElementType(Type t) {
  if (isa<PredType>(t))
    return IntegerType::get(t.getContext(), 1);
  if (t.isIntOrIndexOrFloat())
    return t;
  return Type();
}

// Resolve the per-lane integer slot for one operand's offset under
// per-axis iter-sym substitution. Used by both the value-outs
// (bijection on `[0, prodPar)`) and the value-ins (gather, allows
// repeats, range check only) paths.
static std::optional<int64_t> resolveOperandSlot(sym::Store &store,
                                                 ExprAttr origOff,
                                                 ArrayRef<StringRef> iterNames,
                                                 ArrayRef<int> vals) {
  sym::ExprHandle subbed =
      substituteIterValues(store, origOff.getValue(), iterNames, vals);
  return sym::getIntegerLiteralValue(subbed);
}

// Fully-unrolled lowering: handles every op with at least one
// value-typed operand (ins or outs). The parallel iter space is
// compile-time-unrolled (no `scf.parallel` — the result lives in a
// single SSA register / lane-indexed extract sequence), per-lane
// extracts feed the body, per-lane composes build the value-typed
// outs. Ptr operands ride the same per-parLane emission with contig-
// group merging on the loads/stores; mixed operand kinds work since
// the loop shape is per-lane either way.
//
// Slot mapping:
//   * Value-typed outs: offset must be a bijection on `[0, prodPar)`
//     (one parLane writes each slot exactly once; the result
//     `vector.from_elements` needs every slot populated).
//   * Value-typed ins: offset must evaluate to an integer slot in
//     `[0, count)` per parLane, where `count` is the operand's
//     compile-time lane count. Gather pattern, repeats are fine —
//     multiple lanes may read the same slot.
// `diagnoseUnsupported` already pinned every value-side offset's free syms
// to iter syms only, so `substituteIterValues` produces a pure
// integer per lane.
//
// `lowerOne` dispatches here when any operand is value-typed; the
// partition path handles all-ptr generics where the `scf.parallel`-
// shaped emission is the right form.
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
  ArrayAttr insOff = op.getInsOffsetsAttr();
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
          decomposeLaneIndex(static_cast<int>(parLane), bounds);
      std::optional<int64_t> slot =
          resolveOperandSlot(store, origOff, iterNames, vals);
      if (!slot || *slot < 0 || *slot >= prodPar || !seen.insert(*slot).second)
        return op.emitOpError("value-outs lowering: outs #")
               << oi << " offset is not a bijection on [0, " << prodPar
               << ") at parLane " << parLane;
      slots.push_back(*slot);
    }
    slotPerOut[oi] = std::move(slots);
  }

  // Per-parLane gather slot for each value-typed ins. No bijection
  // (multiple lanes may read the same slot); only the integer-eval +
  // in-range check is required. Skipped for ptr-typed ins (handled
  // by `emitInsLoadsLaned`'s contig-group loop downstream).
  SmallVector<SmallVector<int64_t>> slotPerIn(numIns);
  for (auto [ii, v] : llvm::enumerate(op.getIns())) {
    if (isa<PtrType>(v.getType()))
      continue;
    auto count = getValueOutLaneCount(v.getType());
    if (!count)
      return op.emitOpError("value-ins lowering: ins #")
             << ii << " has no compile-time lane count";
    ExprAttr origOff = getOperandOffset(insOff, ii);
    SmallVector<int64_t> slots;
    slots.reserve(prodPar);
    for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
      SmallVector<int, 4> vals =
          decomposeLaneIndex(static_cast<int>(parLane), bounds);
      std::optional<int64_t> slot =
          resolveOperandSlot(store, origOff, iterNames, vals);
      if (!slot || *slot < 0 || *slot >= *count)
        return op.emitOpError("value-ins lowering: ins #")
               << ii << " offset at parLane " << parLane
               << " does not evaluate to a slot in [0, " << *count << ")";
      slots.push_back(*slot);
    }
    slotPerIn[ii] = std::move(slots);
  }

  // Reuse the partition emission machinery with `p[a] = bounds[a]`
  // on every axis (full unroll) and declaration order. The ins-load
  // path then runs `findContigGroups` over the unrolled lane offsets
  // and merges contiguous runs into vector loads (a 16x16 contiguous
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

  // Value-typed ins: gather pattern. UCC the carrier to builtin
  // `vector<NxBuiltinElem>` (with `!hc.pred` mapped to `i1` — see
  // `asBuiltinElementType`) and emit one `vector.extract` per parLane
  // at the slot the offset evaluates to. Restore the body's expected
  // element type via a trailing UCC so the per-lane mapping in
  // `cloneBody` doesn't shadow it with a non-matching SSA type.
  for (auto [ii, v] : llvm::enumerate(op.getIns())) {
    if (isa<PtrType>(v.getType()))
      continue;
    auto count = getValueOutLaneCount(v.getType());
    auto shaped = cast<SymbolicallyShapedTypeInterface>(v.getType());
    Type elemTy = shaped.getSymbolicElementType();
    Type builtinElem = asBuiltinElementType(elemTy);
    if (!builtinElem)
      return op.emitOpError("value-ins lowering: ins #")
             << ii << " has unsupported element type for vector carrier";
    mlir::VectorType vecTy = mlir::VectorType::get({*count}, builtinElem);
    Value asVec = v;
    if (v.getType() != Type(vecTy))
      asVec = UnrealizedConversionCastOp::create(builder, loc, vecTy, v)
                  .getResult(0);
    insLanes[ii].assign(prodPar, Value());
    for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
      int64_t slot = slotPerIn[ii][parLane];
      Value lane = vector::ExtractOp::create(builder, loc, asVec,
                                             ArrayRef<int64_t>{slot})
                       .getResult();
      if (lane.getType() != elemTy)
        lane = UnrealizedConversionCastOp::create(builder, loc, elemTy, lane)
                   .getResult(0);
      insLanes[ii][parLane] = lane;
    }
  }

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
    // it's not necessarily `parLane` (the offset might permute). Non-
    // builtin element types (`!hc.pred` from a bare-mask out — the
    // shape the `hc.load_mask` rewrite produces) ride the same UCC
    // chain the value-ins path uses: bridge the carrier to its
    // `vector<NxBuiltin>` shape, extract per lane, then UCC each
    // lane back to the body's expected element type so the per-lane
    // mapping in `cloneBody` sees the matching SSA type.
    auto count = getValueOutLaneCount(v.getType());
    auto shaped = cast<SymbolicallyShapedTypeInterface>(v.getType());
    Type elemTy = shaped.getSymbolicElementType();
    Type builtinElem = asBuiltinElementType(elemTy);
    if (!builtinElem)
      return op.emitOpError("value-outs lowering: outs #")
             << oi << " has unsupported element type for vector carrier";
    mlir::VectorType vecTy = mlir::VectorType::get({*count}, builtinElem);
    Value asVec = v;
    if (v.getType() != Type(vecTy))
      asVec = UnrealizedConversionCastOp::create(builder, loc, vecTy, v)
                  .getResult(0);
    for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
      int64_t slot = slotPerOut[oi][parLane];
      Value lane = vector::ExtractOp::create(builder, loc, asVec,
                                             ArrayRef<int64_t>{slot})
                       .getResult();
      if (lane.getType() != elemTy)
        lane = UnrealizedConversionCastOp::create(builder, loc, elemTy, lane)
                   .getResult(0);
      outsInit[parLane][oi] = lane;
    }
  }

  // Per-parLane body clones. The full unroll gives every iter sym a
  // compile-time integer value on each lane, so we materialise a
  // constant-index SSA per (iter_sym, parLane) and feed it as the
  // body's iter-sym scope. `bindIterSymsInClone` walks each cloned
  // `hc.idx_apply` / `hc.pred_apply` in the body and appends an iter
  // sym binding wherever the body's expression references one without
  // an explicit operand — the body-authoring convention the mask
  // emitters rely on (see `bindIterSymsInClone`'s comment).
  SmallVector<SmallVector<Value>> finals(prodPar, SmallVector<Value>(numOuts));
  for (int64_t lane = 0; lane < prodPar; ++lane) {
    SmallVector<Value> insVals(numIns);
    for (size_t ii = 0; ii < numIns; ++ii)
      insVals[ii] = insLanes[ii][lane];
    SmallVector<int, 4> coords =
        decomposeLaneIndex(static_cast<int>(lane), bounds);
    // Start from the ambient scope (`$WG*`, `$WI*`, `$WGS*`, kernel-arg
    // shape syms, ancestor structured-loop induction vars) so body
    // applies that reference any of those as free symbols get explicit
    // operand bindings here, and stamp the per-lane iter sym constants
    // on top to override the zero placeholders `seedAmbientScope` left
    // behind. The post-flatten second `hc-lower-launch-body` pass
    // would otherwise have nothing to look these up against once
    // generic unrolling lifts the body out from under its launch
    // ancestor walker.
    llvm::StringMap<Value> laneIterScope = scope;
    for (auto [ax, c] : llvm::zip_equal(axes, coords))
      laneIterScope[ax.name] =
          arith::ConstantIndexOp::create(builder, loc, c).getResult();
    SmallVector<Value> yielded;
    if (failed(cloneBody(builder, op, insVals, outsInit[lane], laneIterScope,
                         yielded)))
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
    // Mirrors the outs-init path's UCC chain: bridge the body's
    // element type (`!hc.pred` for the mask carrier; arbitrary
    // `!hc.idx<...>` would also fall here) to its builtin element so
    // `vector.from_elements` can compose, then UCC the assembled
    // `vector<NxBuiltin>` back to the operand's bare-vector carrier
    // for the original consumer.
    auto count = getValueOutLaneCount(v.getType());
    auto shaped = cast<SymbolicallyShapedTypeInterface>(v.getType());
    Type elemTy = shaped.getSymbolicElementType();
    Type builtinElem = asBuiltinElementType(elemTy);
    if (!builtinElem)
      return op.emitOpError("value-outs lowering: outs #")
             << oi << " has unsupported element type for vector carrier";
    mlir::VectorType vecTy = mlir::VectorType::get({*count}, builtinElem);
    SmallVector<Value> elems(*count);
    for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
      Value lane = finals[parLane][oi];
      if (lane.getType() != builtinElem)
        lane =
            UnrealizedConversionCastOp::create(builder, loc, builtinElem, lane)
                .getResult(0);
      elems[slotPerOut[oi][parLane]] = lane;
    }
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
// Dispatch order is significant:
//   * Collective first — workgroup-shared outs (direct
//     `!hc.ptr<workgroup>` or `bare_tensor` UCC-backed by one) inside
//     a `gpu.launch` route to `lowerCollective`. Per-lane unroll
//     against shared state would have every thread fight every other
//     thread for every element, so a `bare_tensor` carrier whose
//     backing storage is LDS has to be partitioned across the wave
//     even though the carrier type would otherwise admit the value-
//     outs path. `lowerCollective` RAUWs its result(s) to the
//     original outs SSA (the bare_tensor view of the same storage).
//   * Value-typed operands next — `lowerValueOuts` compile-time-
//     unrolls the parallel sweep so a single SSA register can hold
//     the result; value-typed ins have only a per-lane
//     `vector.extract` materialization. Stand-alone `bare_tensor`
//     outs (no `gpu.launch` ancestor, so no backing LDS) reach this
//     path and ride the same compose-and-UCC-back shape as
//     `bare_vector` outs.
//   * Otherwise — the partition-aware emitter handles all-ptr
//     generics with no workgroup-shared outs.
static LogicalResult lowerOne(HCGenericOp op) {
  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  SmallVector<IterAxis> axes = collectIterAxes(op);

  if (isCollectiveCandidate(op, axes)) {
    if (failed(lowerCollective(op, axes)))
      return failure();
    op.erase();
    return success();
  }

  if (hasValueOuts(op) || hasValueIns(op)) {
    if (failed(lowerValueOuts(op, axes)))
      return failure();
    op.erase();
    return success();
  }

  auto [order, p] = selectBest(op, axes, store);
  if (failed(lowerWithPartition(op, axes, order, p)))
    return failure();

  // The partition path is the all-ptr-out branch — those produce no
  // SSA results, so erasing is sufficient. The belt-and-suspenders
  // error keeps a future relaxation of the candidate gates from
  // silently losing results on this path.
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
      if (diagnoseUnsupported(op))
        continue;
      if (failed(lowerOne(op))) {
        signalPassFailure();
        return;
      }
    }

    // Post-walk: anything still here is a hard failure. A surviving
    // `hc.generic` flows through GPU outlining + ROCDL attach and
    // trips later passes with diagnostics far from the source —
    // diagnose at the production site instead. We re-run the gate
    // here (and not against the originally-rejected set) so that a
    // future `lowerOne` path that mistakenly returns success without
    // erasing also surfaces as a failure rather than silent data
    // loss.
    bool sawSurvivor = false;
    getOperation()->walk([&](HCGenericOp op) {
      auto reason = diagnoseUnsupported(op);
      InFlightDiagnostic diag =
          op.emitError("hc-lower-generic: cannot lower hc.generic");
      if (reason)
        diag << "; " << *reason;
      else
        diag << "; op passed every candidate gate but no lowering path "
                "erased it (internal)";
      sawSurvivor = true;
    });
    if (sawSurvivor)
      signalPassFailure();
  }
};

} // namespace

// `createHCLowerGenericPass()` is emitted by tablegen.
