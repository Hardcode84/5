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
// Pretty-print `t` to a `std::string` for embedding in diagnostic
// messages. Existing call sites already pulled in
// `llvm::raw_string_ostream`; this helper just centralises the
// boilerplate.
static std::string formatType(Type t) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  t.print(os);
  return buf;
}

// `!hc.ptr<addr, element>` with a non-null element type. The address
// space slot can stay generic; what we need is a real pointee for
// the boundary load/store.
static bool isValidPtrOperand(Value v) {
  auto p = dyn_cast<PtrType>(v.getType());
  return p && p.getElementType();
}

// Format the shared ins/outs operand-type error for malformed
// non-ptr operands (anything that isn't a `!hc.ptr` and isn't a
// rank-1 fixed-lane carrier with an integer-literal shape).
static std::string formatBadOperandTypeError(StringRef kind, size_t idx,
                                             Type ty) {
  return (Twine(kind) + " #" + Twine(idx) + " has type '" + formatType(ty) +
          "'; expected !hc.ptr<...> with element type or a rank-1 "
          "fixed-lane carrier with integer-literal shape")
      .str();
}

// Value-typed ins/outs operands are accepted when the carrier
// resolves to a compile-time lane count (rank-1, integer-literal
// shape — same gate the outs side uses). The actual per-lane slot
// evaluation happens in `lowerValueOuts`; the candidate gate just
// admits the op into the fully-unrolled path.
//
// `hasValIn` / `hasValOut` are set to true iff any non-ptr operand
// of the corresponding side was admitted.
static std::optional<std::string>
diagnoseGenericOperandTypes(HCGenericOp op, bool &hasValIn, bool &hasValOut) {
  hasValIn = false;
  for (auto [i, v] : llvm::enumerate(op.getIns())) {
    if (isValidPtrOperand(v))
      continue;
    if (!getValueOutLaneCount(v.getType()))
      return formatBadOperandTypeError("ins", i, v.getType());
    hasValIn = true;
  }
  hasValOut = false;
  for (auto [i, v] : llvm::enumerate(op.getOuts())) {
    if (isValidPtrOperand(v))
      continue;
    if (!getValueOutLaneCount(v.getType()))
      return formatBadOperandTypeError("outs", i, v.getType());
    hasValOut = true;
  }
  return std::nullopt;
}

// No rank-N offset check here — the op verifier already pins
// per-operand offset arity to the operand's rank (ptr → 1, rank-N
// shaped → N), and the operand-type gate above rejects every rank-N
// shaped type that would let a rank-N offset slip through. The
// combination "rank-1 operand, rank-N offsets" is not representable
// in textual IR. A future verifier relaxation should restore the
// check (and pair it with a LIT fixture that actually exercises it).
static std::optional<std::string> diagnoseUndefIterBounds(HCGenericOp op) {
  for (auto [i, bound] : llvm::enumerate(op.getIterBounds()))
    if (auto def = bound.getDefiningOp())
      if (isa<HCUndefValueOp>(def))
        return (Twine("iter #") + Twine(i) +
                " bound is hc.undef_value; bounds must resolve to a "
                "concrete index value")
            .str();
  return std::nullopt;
}

static std::optional<std::string> diagnoseNoNestedScfIf(HCGenericOp op) {
  for (Operation &nested : op.getBody().front())
    if (isa<scf::IfOp>(nested))
      return std::string(
          "body contains nested scf.if; control flow inside the body is not "
          "modeled (lower predicates to hc.yield_predicated first)");
  return std::nullopt;
}

// Fully-unrolled-path bound check: each iter bound must resolve to a
// compile-time integer (either an `arith.constant` index value or an
// `!hc.idx<"<int>">`-typed SSA bound — the frontend planting
// `hc.const` against an integer-only shape is the canonical case).
static std::optional<int64_t> constantBoundValue(Value bound) {
  if (auto def = bound.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(def.getValue()))
      return attr.getInt();
  if (auto idxTy = dyn_cast<IdxType>(bound.getType()))
    if (ExprAttr e = idxTy.getExpr())
      return sym::getIntegerLiteralValue(e.getValue());
  return std::nullopt;
}

static std::optional<std::string> diagnoseConstantIterBounds(HCGenericOp op) {
  for (auto [i, bound] : llvm::enumerate(op.getIterBounds())) {
    std::optional<int64_t> v = constantBoundValue(bound);
    if (!v || *v < 0)
      return (Twine("iter #") + Twine(i) +
              " bound is not a compile-time non-negative integer literal "
              "(value-typed operand needs constant bound)")
          .str();
  }
  return std::nullopt;
}

// All iters must be parallel — the unrolled emitter threads every
// parLane through the result vector / store sequence, and a
// reduction iter would need cross-lane carry that the boundary form
// doesn't model. Value-typed ins reuse the same constraint: the
// gather slot is a function of iter syms alone, and a reduction iter
// would mean the same value-in lane gets read at different reduction
// steps with no scf-loop carry to express it.
static std::optional<std::string> diagnoseAllParallelIters(HCGenericOp op) {
  for (auto [i, k] : llvm::enumerate(op.getIterKindsAttr()))
    if (cast<IterKindAttr>(k).getValue() != IterKind::Parallel)
      return (Twine("iter #") + Twine(i) +
              " kind is reduction (value-typed operand needs all-parallel "
              "iters)")
          .str();
  return std::nullopt;
}

// Body terminator: `hc.yield` (unconditional publish) or
// `hc.yield_predicated` (per-value mask gate → `arith.select` at
// the boundary in `cloneBody`). Anything else escaped the emitters
// we know about.
static std::optional<std::string> diagnoseGenericTerminator(HCGenericOp op) {
  Operation &term = op.getBody().front().back();
  if (!isa<HCYieldOp, HCYieldPredicatedOp>(&term))
    return (Twine("body terminator '") + term.getName().getStringRef() +
            "' is not hc.yield or hc.yield_predicated")
        .str();
  return std::nullopt;
}

// Outs offset's free syms must be a subset of iter syms; ambient
// syms (shape / stride params) would make slot evaluation
// ambient-dependent. Value-typed ins offsets are the per-lane slot
// expressions and follow the same constraint — `lowerValueOuts`
// constant-evaluates them at every parallel-lane combo.
//
// Ptr operands carry their offset against the source pointer and
// may reference ambient syms (the `emitOffset` path resolves them
// via `loopScope` + ambient bindings); only the value-typed operands
// need slot-eval, and slot-eval needs iter-only.
static std::optional<std::string>
findAmbientSymInOffsets(ArrayAttr arr, OperandRange operands, StringRef kind,
                        const llvm::StringSet<> &iterNames) {
  for (size_t i = 0, e = arr.size(); i < e; ++i) {
    Value v = operands[i];
    if (isa<PtrType>(v.getType()))
      continue;
    auto off = cast<ExprAttr>(cast<ArrayAttr>(arr[i])[0]);
    std::optional<std::string> ambient;
    sym::walkSymbolNames(off.getValue(), [&](StringRef name) {
      if (!iterNames.contains(name) && !ambient)
        ambient = name.str();
    });
    if (ambient)
      return (Twine(kind) + " #" + Twine(i) +
              " offset references non-iter symbol '" + *ambient +
              "' (value-typed operand needs iter-only offsets)")
          .str();
  }
  return std::nullopt;
}

static std::optional<std::string> diagnoseAmbientOffsetSyms(HCGenericOp op) {
  llvm::StringSet<> iterNames;
  for (Attribute s : op.getIterSymsAttr())
    iterNames.insert(cast<StringAttr>(s).getValue());
  if (auto r = findAmbientSymInOffsets(op.getInsOffsetsAttr(), op.getIns(),
                                       "ins", iterNames))
    return r;
  return findAmbientSymInOffsets(op.getOutsOffsetsAttr(), op.getOuts(), "outs",
                                 iterNames);
}

static std::optional<std::string> diagnoseUnsupported(HCGenericOp op) {
  bool hasValIn = false;
  bool hasValOut = false;
  if (auto r = diagnoseGenericOperandTypes(op, hasValIn, hasValOut))
    return r;
  if (auto r = diagnoseUndefIterBounds(op))
    return r;
  if (auto r = diagnoseNoNestedScfIf(op))
    return r;
  if (!hasValOut && !hasValIn)
    return std::nullopt;
  if (auto r = diagnoseConstantIterBounds(op))
    return r;
  if (auto r = diagnoseAllParallelIters(op))
    return r;
  if (auto r = diagnoseGenericTerminator(op))
    return r;
  return diagnoseAmbientOffsetSyms(op);
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
namespace {
// Classified view of an apply op for `bindIterSymsInClone`.
// `isIdx` distinguishes `hc.idx_apply` (true) from `hc.pred_apply`
// (false); `existingSyms` is the apply's current `symbols` operand
// names. A nullopt return from `classifyApplyOp` means the op isn't
// an apply we care about, or its result type lacks a payload (no
// expression / predicate to walk).
struct ApplyOpInfo {
  bool isIdx;
  ArrayAttr existingSyms;
};
} // namespace

static std::optional<ApplyOpInfo> classifyApplyOp(Operation *cloned) {
  if (auto idx = dyn_cast<HCIdxApplyOp>(cloned)) {
    auto idxTy = dyn_cast<IdxType>(idx.getResult().getType());
    if (!idxTy || !idxTy.getExpr())
      return std::nullopt;
    return ApplyOpInfo{true, idx.getSymbolsAttr()};
  }
  if (auto pred = dyn_cast<HCPredApplyOp>(cloned)) {
    auto predTy = dyn_cast<PredType>(pred.getResult().getType());
    if (!predTy || !predTy.getPred())
      return std::nullopt;
    return ApplyOpInfo{false, pred.getSymbolsAttr()};
  }
  return std::nullopt;
}

// Walk the cloned apply op's payload (idx expression or pred
// predicate) and collect every iter-scope-bound sym that isn't
// already in `existingSyms`. The output preserves first-seen order
// and de-duplicates against itself; the caller sorts for determinism.
static void collectMissingIterSyms(Operation *cloned, bool isIdx,
                                   ArrayAttr existingSyms,
                                   const llvm::StringMap<Value> &iterScope,
                                   SmallVectorImpl<StringRef> &additions) {
  llvm::StringSet<> already;
  for (Attribute n : existingSyms)
    already.insert(cast<StringAttr>(n).getValue());

  auto walker = [&](StringRef name) {
    if (already.contains(name) || !iterScope.count(name))
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
}

// Rebuild the apply op with the additional iter syms appended to its
// `symbols` operand list (and their matching SSA values pulled out
// of `iterScope`). Caller has already deduped `additions` and sorted
// it for deterministic textual form.
static Operation *
createApplyOpWithIterSyms(OpBuilder &builder, Operation *cloned, bool isIdx,
                          ArrayRef<StringRef> additions, ArrayAttr existingSyms,
                          const llvm::StringMap<Value> &iterScope) {
  SmallVector<Value> newOperands(cloned->getOperands());
  SmallVector<Attribute> newSyms(existingSyms.begin(), existingSyms.end());
  for (StringRef name : additions) {
    newOperands.push_back(iterScope.lookup(name));
    newSyms.push_back(builder.getStringAttr(name));
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(cloned);
  if (isIdx)
    return HCIdxApplyOp::create(builder, cloned->getLoc(),
                                cloned->getResult(0).getType(), newOperands,
                                builder.getArrayAttr(newSyms));
  return HCPredApplyOp::create(builder, cloned->getLoc(),
                               cloned->getResult(0).getType(), newOperands,
                               builder.getArrayAttr(newSyms));
}

static void bindIterSymsInClone(OpBuilder &builder, Operation &nested,
                                Operation *cloned,
                                const llvm::StringMap<Value> &iterScope,
                                IRMapping &mapping) {
  if (iterScope.empty())
    return;
  std::optional<ApplyOpInfo> info = classifyApplyOp(cloned);
  if (!info)
    return;

  SmallVector<StringRef, 4> additions;
  collectMissingIterSyms(cloned, info->isIdx, info->existingSyms, iterScope,
                         additions);
  if (additions.empty())
    return;

  // Sort for deterministic textual form across rebuilds — see the
  // matching note on `emitOffset`.
  llvm::sort(additions);
  Operation *replacement = createApplyOpWithIterSyms(
      builder, cloned, info->isIdx, additions, info->existingSyms, iterScope);
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
// Apply the per-lane `hc.yield_predicated` to fold the masked
// per-value result against its `outs` init. Each lane's yielded
// value is selected against the lane's init via `arith.select`; the
// mask is coerced to `i1` so it matches `arith.select`'s signature.
static LogicalResult
cloneBodyPredicatedYield(OpBuilder &builder, HCGenericOp op,
                         HCYieldPredicatedOp pyield, ValueRange outsVals,
                         const IRMapping &mapping,
                         SmallVectorImpl<Value> &yieldedOut) {
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
    Value sel =
        arith::SelectOp::create(builder, loc, i1Mask, mapped, init).getResult();
    yieldedOut.push_back(sel);
  }
  return success();
}

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
  if (auto pyield = dyn_cast<HCYieldPredicatedOp>(&term))
    return cloneBodyPredicatedYield(builder, op, pyield, outsVals, mapping,
                                    yieldedOut);
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

// Clone the body for one lane: pick out the lane's ins, hand it the
// current accumulator slot for its parallel-lane, run the body, and
// stash the yielded values back into the accumulator. Partition /
// reduction path doesn't expose iter-sym SSA bindings to the body —
// `emitInsLoadsLaned` bakes deltas into the operand-offset side, and
// the body's only iter-sym story today is the value-outs unroll in
// `lowerValueOuts` — so we pass an empty iter scope and
// `bindIterSymsInClone` no-ops. Wiring this up properly is a
// follow-up if a body op ever references an iter sym from the
// partition path.
static LogicalResult emitOneLaneBody(OpBuilder &builder, HCGenericOp op,
                                     int lane, ArrayRef<size_t> order,
                                     ArrayRef<size_t> parOrder, ArrayRef<int> p,
                                     ArrayRef<SmallVector<Value>> insLanes,
                                     size_t numOuts,
                                     MutableArrayRef<SmallVector<Value>> acc) {
  SmallVector<int, 4> delta = decomposeLane(lane, order, p);
  int parLane = computeSubLane(delta, parOrder, p);
  SmallVector<Value> insVals(insLanes.size());
  for (size_t ii = 0; ii < insLanes.size(); ++ii)
    insVals[ii] = insLanes[ii][lane];
  SmallVector<Value> outsVals = acc[parLane];
  SmallVector<Value> yielded;
  llvm::StringMap<Value> emptyIterScope;
  if (failed(
          cloneBody(builder, op, insVals, outsVals, emptyIterScope, yielded)))
    return failure();
  if (yielded.size() != numOuts)
    return op.emitOpError("body yielded wrong arity");
  acc[parLane] = std::move(yielded);
  return success();
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

  for (int lane = 0; lane < prodAll; ++lane)
    if (failed(emitOneLaneBody(builder, op, lane, order, parOrder, p, insLanes,
                               numOuts, acc)))
      return failure();

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

// Lower one parallel-iter invocation. Loads outs initial values,
// either emits the body directly (when there are no reduction axes)
// or wraps an scf.for reduction nest, then stores the finals.
static LogicalResult lowerPartitionPerParallelBody(
    OpBuilder &builder, HCGenericOp op, ArrayRef<IterAxis> axes,
    ArrayRef<size_t> order, ArrayRef<int> p, ArrayRef<size_t> redOrder,
    llvm::StringMap<Value> &scope, sym::Store &store) {
  SmallVector<Value> initOuts =
      emitOutsInitLoadsPartitioned(builder, op, axes, order, p, scope, store);
  SmallVector<Value> finals;
  if (redOrder.empty()) {
    auto yielded = emitInnerBodyClones(builder, op, axes, order, p, initOuts,
                                       scope, store);
    if (failed(yielded))
      return failure();
    finals = std::move(*yielded);
  } else {
    auto reduced = emitReductionNestPartitioned(
        builder, op, axes, order, p, redOrder, initOuts, scope, store);
    if (failed(reduced))
      return failure();
    finals = std::move(*reduced);
  }
  emitOutsStoresPartitioned(builder, op, axes, order, p, finals, scope, store);
  return success();
}

// Per-parallel-axis ParallelOp bounds: zero lower, dim upper,
// partition step `p[a]`. Fills the three output vectors in axis
// order matching `parIdx`.
static void buildPartitionParallelBounds(
    OpBuilder &builder, Location loc, ValueRange bounds,
    ArrayRef<size_t> parIdx, ArrayRef<int> p, SmallVectorImpl<Value> &lowers,
    SmallVectorImpl<Value> &uppers, SmallVectorImpl<Value> &steps) {
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  lowers.assign(parIdx.size(), c0);
  uppers.reserve(parIdx.size());
  steps.reserve(parIdx.size());
  for (size_t pi : parIdx) {
    uppers.push_back(castIdxToIndex(builder, loc, bounds[pi]));
    steps.push_back(
        arith::ConstantIndexOp::create(builder, loc, p[pi]).getResult());
  }
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

  if (parIdx.empty()) {
    llvm::StringMap<Value> scope;
    seedAmbientScope(op, scope);
    return lowerPartitionPerParallelBody(builder, op, axes, order, p, redOrder,
                                         scope, store);
  }

  ValueRange bounds = op.getIterBounds();
  SmallVector<Value> lowers, uppers, steps;
  buildPartitionParallelBounds(builder, loc, bounds, parIdx, p, lowers, uppers,
                               steps);

  LogicalResult bodyStatus = success();
  scf::ParallelOp::create(builder, loc, lowers, uppers, steps,
                          [&](OpBuilder &b, Location, ValueRange ivs) {
                            llvm::StringMap<Value> scope;
                            seedAmbientScope(op, scope);
                            for (auto [k, pi] : llvm::enumerate(parIdx))
                              scope[axes[pi].name] = ivs[k];
                            bodyStatus = lowerPartitionPerParallelBody(
                                b, op, axes, order, p, redOrder, scope, store);
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

// Pin every outs to a workgroup ptr we can ptr_offset/load/store
// against. Direct ptr outs are themselves; bare-tensor outs UCC
// through to their backing ptr<workgroup> type once, hoisted above
// the chunk loop so the cast doesn't re-emit per chunk. The upstream
// UCC `ptr<workgroup> → bare_tensor` (planted by
// `hc-lower-launch-body`'s shaped-constant lowering on the producing
// side) and this fresh `bare_tensor → ptr<workgroup>` form a
// foldable pair: canonicalize collapses the chain back to the
// original `hc.alloc workgroup` so per-thread stores hit the real
// LDS storage without a UCC dead-end at LLVM translation.
static SmallVector<Value> collectiveOutsPtrs(OpBuilder &builder, Location loc,
                                             HCGenericOp op) {
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
  return outsPtrs;
}

// Total iter-space cardinality and per-chunk count for collective
// dispatch. `total = prod(bounds)`, `chunks = ceil(total / wgSize)`.
static std::pair<Value, Value> collectiveTotalAndChunks(OpBuilder &builder,
                                                        Location loc,
                                                        ValueRange bounds,
                                                        Value wgSize) {
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  Value total = c1;
  for (Value bound : bounds) {
    Value dim = castIdxToIndex(builder, loc, bound);
    total = arith::MulIOp::create(builder, loc, total, dim).getResult();
  }
  Value chunks =
      arith::CeilDivUIOp::create(builder, loc, total, wgSize).getResult();
  return {total, chunks};
}

// Unlinearise `lin` into per-axis coords: rightmost axis varies
// fastest, matching the flatten convention used by
// `hc-flatten-with-layouts` so post-flatten offset expressions line
// up with the coord assignment.
static SmallVector<Value> unlinearizeCollectiveCoords(OpBuilder &builder,
                                                      Location loc, Value lin,
                                                      ValueRange bounds) {
  SmallVector<Value> coords(bounds.size());
  Value remaining = lin;
  for (int axis = static_cast<int>(bounds.size()) - 1; axis >= 0; --axis) {
    Value dim = castIdxToIndex(builder, loc, bounds[axis]);
    coords[axis] =
        arith::RemUIOp::create(builder, loc, remaining, dim).getResult();
    if (axis > 0)
      remaining =
          arith::DivUIOp::create(builder, loc, remaining, dim).getResult();
  }
  return coords;
}

// Load one operand's element at `off` against the ptr at `ptr`,
// then bridge the loaded value back to `bodyArgTy` if the body's
// block arg keeps a symbolic element type that differs from the
// builtin surrogate (`!hc.pred` body arg ↔ `i1` LDS storage etc.)
static Value loadCollectiveOperandElement(OpBuilder &builder, Location loc,
                                          Value ptr, Value off,
                                          Type bodyArgTy) {
  Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
  Value addr =
      HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off).getResult();
  Value loaded = HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult();
  if (loaded.getType() != bodyArgTy)
    loaded = UnrealizedConversionCastOp::create(builder, loc, bodyArgTy, loaded)
                 .getResult(0);
  return loaded;
}

// Store one yielded scalar back at `off` against the ptr at `ptr`,
// inserting the symmetric UCC bridge if the body's element type
// differs from the LDS pointer's surrogate.
static void storeCollectiveYielded(OpBuilder &builder, Location loc, Value ptr,
                                   Value off, Value yielded) {
  Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
  Value addr =
      HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off).getResult();
  Value toStore = yielded;
  if (toStore.getType() != elemTy)
    toStore = UnrealizedConversionCastOp::create(builder, loc, elemTy, toStore)
                  .getResult(0);
  HCPtrStoreOp::create(builder, loc, toStore, addr);
}

// Load every ins / outs operand at its composed offset under
// `scope`. The result vectors are parallel to the op's ins / outs.
static void loadCollectiveOperands(OpBuilder &builder, Location loc,
                                   HCGenericOp op, ArrayRef<Value> outsPtrs,
                                   const llvm::StringMap<Value> &scope,
                                   SmallVectorImpl<Value> &insVals,
                                   SmallVectorImpl<Value> &outsVals) {
  ArrayAttr insOff = op.getInsOffsetsAttr();
  insVals.resize(op.getIns().size());
  for (size_t ii = 0; ii < op.getIns().size(); ++ii) {
    Value off = emitOffset(builder, loc, getOperandOffset(insOff, ii), scope);
    Type bodyArgTy = op.getBody().front().getArgument(ii).getType();
    insVals[ii] = loadCollectiveOperandElement(builder, loc, op.getIns()[ii],
                                               off, bodyArgTy);
  }
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  outsVals.resize(op.getOuts().size());
  for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
    Value off = emitOffset(builder, loc, getOperandOffset(outsOff, oi), scope);
    Type bodyArgTy =
        op.getBody().front().getArgument(op.getIns().size() + oi).getType();
    outsVals[oi] = loadCollectiveOperandElement(builder, loc, outsPtrs[oi], off,
                                                bodyArgTy);
  }
}

// Clone the body for one in-range chunk-thread and store the yielded
// values back at each outs ptr's composed offset. The `scope` map
// already mixes ambient and iter sym SSA per the offset-emission
// setup above; `bindIterSymsInClone` only acts on names the apply's
// pred/expr actually references and that aren't already in the
// apply's `symbols` list, so any unused entries are harmless.
static LogicalResult
emitCollectiveChunkInRange(OpBuilder &builder, Location loc, HCGenericOp op,
                           ArrayRef<Value> outsPtrs, ValueRange insVals,
                           ValueRange outsVals,
                           const llvm::StringMap<Value> &scope) {
  SmallVector<Value> yielded;
  if (failed(cloneBody(builder, op, insVals, outsVals, scope, yielded)))
    return failure();
  if (yielded.size() != op.getOuts().size())
    return op.emitOpError("body yielded wrong arity");
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
    Value off = emitOffset(builder, loc, getOperandOffset(outsOff, oi), scope);
    storeCollectiveYielded(builder, loc, outsPtrs[oi], off, yielded[oi]);
  }
  return success();
}

// One chunk-thread's body inside the `lin < total` guard. Builds the
// iter-sym scope from the unlinearised coords, loads every operand at
// its composed offset, runs the body, and stores yielded values back.
static LogicalResult emitCollectiveChunkBody(OpBuilder &builder, Location loc,
                                             HCGenericOp op,
                                             ArrayRef<IterAxis> axes,
                                             ValueRange bounds, Value lin,
                                             ArrayRef<Value> outsPtrs) {
  SmallVector<Value> coords =
      unlinearizeCollectiveCoords(builder, loc, lin, bounds);
  llvm::StringMap<Value> scope;
  seedAmbientScope(op, scope);
  for (auto [ax, coord] : llvm::zip(axes, coords))
    scope[ax.name] = coord;

  SmallVector<Value> insVals, outsVals;
  loadCollectiveOperands(builder, loc, op, outsPtrs, scope, insVals, outsVals);
  return emitCollectiveChunkInRange(builder, loc, op, outsPtrs, insVals,
                                    outsVals, scope);
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
  auto [total, chunks] = collectiveTotalAndChunks(builder, loc, bounds, wgSize);
  SmallVector<Value> outsPtrs = collectiveOutsPtrs(builder, loc, op);

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
    OpBuilder::InsertionGuard rangeGuard(builder);
    builder.setInsertionPointToStart(&rangeIf.getThenRegion().front());
    bodyStatus =
        emitCollectiveChunkBody(builder, loc, op, axes, bounds, lin, outsPtrs);
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
namespace {
// Bundle of compile-time integer parameters threaded through the
// value-outs lowering: the per-axis bound, the iter-sym names, and
// the parallel lane count `prodPar = prod(bounds)`.
struct ValueOutsLanes {
  SmallVector<int64_t> bounds;
  SmallVector<StringRef> iterNames;
  int64_t prodPar = 1;
};

// `vector<NxBuiltin>` carrier we UCC value-typed operands through:
// the canonical bare-vector form `vector.extract` /
// `vector.from_elements` accept. `elemTy` is the operand's symbolic
// element type (e.g. `!hc.pred`); `builtinElem` is its surrogate
// (e.g. `i1`); `vecTy` is `vector<count x builtinElem>`.
struct VectorCarrier {
  Type elemTy;
  Type builtinElem;
  mlir::VectorType vecTy;
};
} // namespace

// Materialise per-axis bounds + iter-sym names + total lane count.
// Returns failure if any iter bound isn't a compile-time integer.
static FailureOr<ValueOutsLanes>
collectValueOutsLanes(HCGenericOp op, ArrayRef<IterAxis> axes) {
  ValueOutsLanes out;
  out.bounds.reserve(axes.size());
  out.iterNames.reserve(axes.size());
  for (const IterAxis &ax : axes) {
    int64_t b = constIterBound(ax.bound);
    if (b < 0)
      return op.emitOpError(
          "value-outs lowering requires constant iter bounds");
    out.bounds.push_back(b);
    out.iterNames.push_back(ax.name);
  }
  out.prodPar = 1;
  for (int64_t b : out.bounds)
    out.prodPar *= b;
  return out;
}

// Verify every value-typed out has a rank-1 integer shape of the
// same width — one slot per parLane. Ptr outs skip the check.
static LogicalResult validateValueOutsLaneCount(HCGenericOp op,
                                                int64_t prodPar) {
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType()))
      continue;
    auto count = getValueOutLaneCount(v.getType());
    if (!count || *count != prodPar)
      return op.emitOpError("value-outs lowering: outs #")
             << oi << " has lane count " << (count ? *count : -1)
             << " but parallel iter space has " << prodPar << " lanes";
  }
  return success();
}

// Per-parLane slot via constant-eval of an operand offset against the
// axes' compile-time bounds. `requireBijection = true` rejects gaps /
// collisions; otherwise duplicates are allowed (gather pattern).
static FailureOr<SmallVector<int64_t>>
resolvePerLaneSlots(HCGenericOp op, sym::Store &store, ExprAttr origOff,
                    const ValueOutsLanes &lanes, int64_t laneCount,
                    bool requireBijection, StringRef errPrefix,
                    size_t operandIdx) {
  SmallVector<int64_t> slots;
  slots.reserve(lanes.prodPar);
  llvm::SmallDenseSet<int64_t, 16> seen;
  for (int64_t parLane = 0; parLane < lanes.prodPar; ++parLane) {
    SmallVector<int, 4> vals =
        decomposeLaneIndex(static_cast<int>(parLane), lanes.bounds);
    std::optional<int64_t> slot =
        resolveOperandSlot(store, origOff, lanes.iterNames, vals);
    bool collision = requireBijection && slot && !seen.insert(*slot).second;
    if (!slot || *slot < 0 || *slot >= laneCount || collision) {
      if (requireBijection)
        return op.emitOpError(errPrefix)
               << operandIdx << " offset is not a bijection on [0, "
               << laneCount << ") at parLane " << parLane;
      return op.emitOpError(errPrefix)
             << operandIdx << " offset at parLane " << parLane
             << " does not evaluate to a slot in [0, " << laneCount << ")";
    }
    slots.push_back(*slot);
  }
  return slots;
}

// Per-operand slot maps. `outs` requires a bijection over [0, prodPar);
// `ins` is a gather (any in-range slot is fine, duplicates allowed).
static FailureOr<SmallVector<SmallVector<int64_t>>>
resolveValueOutsSlots(HCGenericOp op, sym::Store &store, ArrayAttr outsOff,
                      const ValueOutsLanes &lanes) {
  SmallVector<SmallVector<int64_t>> slotPerOut(op.getOuts().size());
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType()))
      continue;
    FailureOr<SmallVector<int64_t>> slots = resolvePerLaneSlots(
        op, store, getOperandOffset(outsOff, oi), lanes, lanes.prodPar,
        /*requireBijection=*/true, "value-outs lowering: outs #", oi);
    if (failed(slots))
      return failure();
    slotPerOut[oi] = std::move(*slots);
  }
  return slotPerOut;
}

static FailureOr<SmallVector<SmallVector<int64_t>>>
resolveValueInsSlots(HCGenericOp op, sym::Store &store, ArrayAttr insOff,
                     const ValueOutsLanes &lanes) {
  SmallVector<SmallVector<int64_t>> slotPerIn(op.getIns().size());
  for (auto [ii, v] : llvm::enumerate(op.getIns())) {
    if (isa<PtrType>(v.getType()))
      continue;
    auto count = getValueOutLaneCount(v.getType());
    if (!count)
      return op.emitOpError("value-ins lowering: ins #")
             << ii << " has no compile-time lane count";
    FailureOr<SmallVector<int64_t>> slots = resolvePerLaneSlots(
        op, store, getOperandOffset(insOff, ii), lanes, *count,
        /*requireBijection=*/false, "value-ins lowering: ins #", ii);
    if (failed(slots))
      return failure();
    slotPerIn[ii] = std::move(*slots);
  }
  return slotPerIn;
}

// Compute the per-operand `vector<NxBuiltin>` carrier shape that
// `vector.extract` / `vector.from_elements` accept. Returns failure
// if the symbolic element type has no builtin surrogate.
static FailureOr<VectorCarrier> valueOperandVectorCarrier(HCGenericOp op,
                                                          Value v,
                                                          StringRef errPrefix,
                                                          size_t operandIdx) {
  auto count = getValueOutLaneCount(v.getType());
  auto shaped = cast<SymbolicallyShapedTypeInterface>(v.getType());
  Type elemTy = shaped.getSymbolicElementType();
  Type builtinElem = asBuiltinElementType(elemTy);
  if (!builtinElem)
    return op.emitOpError(errPrefix)
           << operandIdx << " has unsupported element type for vector carrier";
  return VectorCarrier{elemTy, builtinElem,
                       mlir::VectorType::get({*count}, builtinElem)};
}

// Trace back through a one-input-one-output UCC to the underlying
// `!hc.ptr<workgroup, T>` that backs a bare-tensor value, matching
// the `ptr<workgroup> -> bare_tensor` source materialization the
// launch-body shaped-constant lowering plants on `hc.zeros`. The
// pair was designed to round-trip through canonicalize when the
// consumer also wants a workgroup ptr; consumers that want a real
// `vector<NxT>` carrier (the value-typed ins/outs path here) instead
// need an actual load chain off the ptr, otherwise the unfoldable
// `ptr -> bare_tensor -> vector` UCC chain survives all the way to
// LLVM translation.
static Value workgroupPtrBackingBareTensor(Value v) {
  auto cast = v.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast.getOutputs().size() != 1)
    return Value();
  Value src = cast.getInputs().front();
  auto ptr = dyn_cast<PtrType>(src.getType());
  if (!ptr || ptr.getAddrSpace() != AddrSpace::Workgroup)
    return Value();
  return src;
}

// Build a `vector<NxBuiltin>` from per-element loads off a workgroup
// LDS ptr. The flat slot layout matches the launch-body
// shaped-constant convention (row-major `ptr + slot` for slot in
// `[0, count)`), so per-lane gathers via `vector.extract` line up
// with the producing `hc.ptr_store` slots the partition lowering
// emitted.
static Value loadVectorFromWorkgroupPtr(OpBuilder &builder, Location loc,
                                        Value ptr, mlir::VectorType vecTy) {
  Type elemTy = vecTy.getElementType();
  PtrType ptrTy = cast<PtrType>(ptr.getType());
  int64_t count = vecTy.getNumElements();
  SmallVector<Value> elems;
  elems.reserve(count);
  for (int64_t slot = 0; slot < count; ++slot) {
    Value off = arith::ConstantIndexOp::create(builder, loc, slot).getResult();
    Value addr =
        HCPtrOffsetOp::create(builder, loc, ptrTy, ptr, off).getResult();
    elems.push_back(
        HCPtrLoadOp::create(builder, loc, elemTy, addr).getResult());
  }
  return vector::FromElementsOp::create(builder, loc, vecTy, elems).getResult();
}

// Materialise `v` as the value-typed lane carrier `vecTy`. The fast
// path is a no-op when `v` is already a vector of the same shape;
// the bare-tensor path threads through `workgroupPtrBackingBareTensor`
// + `loadVectorFromWorkgroupPtr` so the per-lane consumers see real
// `hc.ptr_load` values instead of an unresolved
// `bare_tensor -> vector` UCC. The last-resort UCC remains for
// non-workgroup carriers (e.g. value-typed outs threaded as-is by an
// upstream op); per-lane extract on those folds elsewhere because
// the bare carrier is itself a vector-shaped SSA.
static Value castToVectorCarrier(OpBuilder &builder, Location loc, Value v,
                                 mlir::VectorType vecTy) {
  if (v.getType() == Type(vecTy))
    return v;
  if (Value ptr = workgroupPtrBackingBareTensor(v))
    return loadVectorFromWorkgroupPtr(builder, loc, ptr, vecTy);
  return UnrealizedConversionCastOp::create(builder, loc, vecTy, v)
      .getResult(0);
}

// One per-lane extraction step: `vector.extract` at compile-time
// slot, then UCC back to the body's expected element type so
// `cloneBody`'s per-lane mapping sees the matching SSA type.
static Value extractAndCoerceLane(OpBuilder &builder, Location loc, Value asVec,
                                  int64_t slot, Type bodyElemTy) {
  Value lane =
      vector::ExtractOp::create(builder, loc, asVec, ArrayRef<int64_t>{slot})
          .getResult();
  if (lane.getType() != bodyElemTy)
    lane = UnrealizedConversionCastOp::create(builder, loc, bodyElemTy, lane)
               .getResult(0);
  return lane;
}

// Value-typed ins: gather pattern. UCC the carrier to its builtin
// vector shape and emit one `vector.extract` per parLane at the slot
// the offset evaluates to. Restore the body's expected element type
// via a trailing UCC so the per-lane mapping in `cloneBody` doesn't
// shadow it with a non-matching SSA type.
static LogicalResult
overrideInsLanesForValueIns(OpBuilder &builder, Location loc, HCGenericOp op,
                            ArrayRef<SmallVector<int64_t>> slotPerIn,
                            int64_t prodPar,
                            SmallVectorImpl<SmallVector<Value>> &insLanes) {
  for (auto [ii, v] : llvm::enumerate(op.getIns())) {
    if (isa<PtrType>(v.getType()))
      continue;
    FailureOr<VectorCarrier> carrier =
        valueOperandVectorCarrier(op, v, "value-ins lowering: ins #", ii);
    if (failed(carrier))
      return failure();
    Value asVec = castToVectorCarrier(builder, loc, v, carrier->vecTy);
    insLanes[ii].assign(prodPar, Value());
    for (int64_t parLane = 0; parLane < prodPar; ++parLane)
      insLanes[ii][parLane] = extractAndCoerceLane(
          builder, loc, asVec, slotPerIn[ii][parLane], carrier->elemTy);
  }
  return success();
}

// Ptr-typed outs init: same per-lane init-load path the partition
// emitter uses, with contig-group merging on the load side.
static void emitPtrOutsInit(OpBuilder &builder, Location loc, HCGenericOp op,
                            size_t oi, Value v, ExprAttr origOff,
                            ArrayRef<IterAxis> axes, ArrayRef<size_t> order,
                            ArrayRef<int> p, ArrayRef<bool> parMask,
                            int64_t prodPar,
                            const llvm::StringMap<Value> &scope,
                            sym::Store &store,
                            MutableArrayRef<SmallVector<Value>> outsInit) {
  Type elemTy = cast<PtrType>(v.getType()).getElementType();
  SmallVector<sym::ExprHandle> offs =
      laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
  auto groups = findContigGroups(store, offs);
  SmallVector<Value> initSlots(prodPar);
  for (const ContigGroup &g : groups) {
    Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes, order,
                               p, parMask, g.start);
    Value addr =
        HCPtrOffsetOp::create(builder, loc, v.getType(), v, off).getResult();
    emitGroupLoad(builder, loc, v, elemTy, addr, g, initSlots);
  }
  for (int64_t parLane = 0; parLane < prodPar; ++parLane)
    outsInit[parLane][oi] = initSlots[parLane];
}

// Value-typed out init: per-parLane `vector.extract` at the offset-
// evaluated slot, with the same UCC chain the value-ins path uses.
static LogicalResult
emitValueOutInit(OpBuilder &builder, Location loc, HCGenericOp op, size_t oi,
                 Value v, ArrayRef<int64_t> slotForOut, int64_t prodPar,
                 MutableArrayRef<SmallVector<Value>> outsInit) {
  FailureOr<VectorCarrier> carrier =
      valueOperandVectorCarrier(op, v, "value-outs lowering: outs #", oi);
  if (failed(carrier))
    return failure();
  Value asVec = castToVectorCarrier(builder, loc, v, carrier->vecTy);
  for (int64_t parLane = 0; parLane < prodPar; ++parLane)
    outsInit[parLane][oi] = extractAndCoerceLane(
        builder, loc, asVec, slotForOut[parLane], carrier->elemTy);
  return success();
}

// Per-parLane outs init: value-typed ones extract from the SSA out
// (via UCC if the carrier isn't builtin `vector`); ptr-typed ones
// ride the same per-lane init-load path the partition emitter uses.
static LogicalResult
emitOutsInit(OpBuilder &builder, Location loc, HCGenericOp op,
             ArrayRef<SmallVector<int64_t>> slotPerOut, ArrayAttr outsOff,
             ArrayRef<IterAxis> axes, ArrayRef<size_t> order, ArrayRef<int> p,
             ArrayRef<bool> parMask, int64_t prodPar,
             const llvm::StringMap<Value> &scope, sym::Store &store,
             MutableArrayRef<SmallVector<Value>> outsInit) {
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType())) {
      emitPtrOutsInit(builder, loc, op, oi, v, getOperandOffset(outsOff, oi),
                      axes, order, p, parMask, prodPar, scope, store, outsInit);
      continue;
    }
    if (failed(emitValueOutInit(builder, loc, op, oi, v, slotPerOut[oi],
                                prodPar, outsInit)))
      return failure();
  }
  return success();
}

// Per-parLane body clones. The full unroll gives every iter sym a
// compile-time integer value on each lane, so we materialise a
// constant-index SSA per (iter_sym, parLane) and feed it as the
// body's iter-sym scope. `bindIterSymsInClone` walks each cloned
// `hc.idx_apply` / `hc.pred_apply` in the body and appends an iter
// sym binding wherever the body's expression references one without
// an explicit operand — the body-authoring convention the mask
// emitters rely on (see `bindIterSymsInClone`'s comment).
//
// Start from the ambient scope (`$WG*`, `$WI*`, `$WGS*`, kernel-arg
// shape syms, ancestor structured-loop induction vars) so body
// applies that reference any of those as free symbols get explicit
// operand bindings here, and stamp the per-lane iter sym constants
// on top to override the zero placeholders `seedAmbientScope` left
// behind. The post-flatten second `hc-lower-launch-body` pass would
// otherwise have nothing to look these up against once generic
// unrolling lifts the body out from under its launch ancestor
// walker.
static LogicalResult emitBodyClonesPerLane(
    OpBuilder &builder, Location loc, HCGenericOp op, ArrayRef<IterAxis> axes,
    const ValueOutsLanes &lanes, ArrayRef<SmallVector<Value>> insLanes,
    ArrayRef<SmallVector<Value>> outsInit, const llvm::StringMap<Value> &scope,
    SmallVectorImpl<SmallVector<Value>> &finals) {
  size_t numIns = op.getIns().size();
  size_t numOuts = op.getOuts().size();
  for (int64_t lane = 0; lane < lanes.prodPar; ++lane) {
    SmallVector<Value> insVals(numIns);
    for (size_t ii = 0; ii < numIns; ++ii)
      insVals[ii] = insLanes[ii][lane];
    SmallVector<int, 4> coords =
        decomposeLaneIndex(static_cast<int>(lane), lanes.bounds);
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
  return success();
}

// Ptr-typed outs ride the same per-parLane store path the partition
// emitter uses, with contig-group merging on the writes.
static void emitPtrOutsStores(OpBuilder &builder, Location loc, HCGenericOp op,
                              size_t oi, Value v, ExprAttr origOff,
                              ArrayRef<IterAxis> axes, ArrayRef<size_t> order,
                              ArrayRef<int> p, ArrayRef<bool> parMask,
                              int64_t prodPar,
                              const llvm::StringMap<Value> &scope,
                              sym::Store &store,
                              ArrayRef<SmallVector<Value>> finals) {
  Type elemTy = cast<PtrType>(v.getType()).getElementType();
  SmallVector<sym::ExprHandle> offs =
      laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
  auto groups = findContigGroups(store, offs);
  SmallVector<Value> laneFinals(prodPar);
  for (int64_t parLane = 0; parLane < prodPar; ++parLane)
    laneFinals[parLane] = finals[parLane][oi];
  for (const ContigGroup &g : groups) {
    Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes, order,
                               p, parMask, g.start);
    Value addr =
        HCPtrOffsetOp::create(builder, loc, v.getType(), v, off).getResult();
    emitGroupStore(builder, loc, elemTy, addr, g, laneFinals);
  }
}

// Value-typed outs compose all finals through one
// `vector.from_elements` sized to the result vector — the slot
// mapping placed each parLane's final at its compile-time slot
// index. Mirrors the outs-init path's UCC chain: bridge the body's
// element type (`!hc.pred` for the mask carrier; arbitrary
// `!hc.idx<...>` would also fall here) to its builtin element so
// `vector.from_elements` can compose, then UCC the assembled
// `vector<NxBuiltin>` back to the operand's bare-vector carrier for
// the original consumer.
static FailureOr<Value>
emitValueOutResult(OpBuilder &builder, Location loc, HCGenericOp op, size_t oi,
                   Value v, ArrayRef<int64_t> slotForOut, int64_t prodPar,
                   ArrayRef<SmallVector<Value>> finals) {
  FailureOr<VectorCarrier> carrier =
      valueOperandVectorCarrier(op, v, "value-outs lowering: outs #", oi);
  if (failed(carrier))
    return failure();
  SmallVector<Value> elems(carrier->vecTy.getNumElements());
  for (int64_t parLane = 0; parLane < prodPar; ++parLane) {
    Value lane = finals[parLane][oi];
    if (lane.getType() != carrier->builtinElem)
      lane = UnrealizedConversionCastOp::create(builder, loc,
                                                carrier->builtinElem, lane)
                 .getResult(0);
    elems[slotForOut[parLane]] = lane;
  }
  Value vec =
      vector::FromElementsOp::create(builder, loc, carrier->vecTy, elems)
          .getResult();
  if (v.getType() == Type(carrier->vecTy))
    return vec;
  return UnrealizedConversionCastOp::create(builder, loc, v.getType(), vec)
      .getResult(0);
}

// Walk every out: emit the ptr-store contig group (no result), or
// compose the value-out vector result into `opResults`.
static LogicalResult
emitOutsFinalize(OpBuilder &builder, Location loc, HCGenericOp op,
                 ArrayRef<SmallVector<int64_t>> slotPerOut, ArrayAttr outsOff,
                 ArrayRef<IterAxis> axes, ArrayRef<size_t> order,
                 ArrayRef<int> p, ArrayRef<bool> parMask, int64_t prodPar,
                 const llvm::StringMap<Value> &scope, sym::Store &store,
                 ArrayRef<SmallVector<Value>> finals,
                 SmallVectorImpl<Value> &opResults) {
  size_t resultIdx = 0;
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType())) {
      emitPtrOutsStores(builder, loc, op, oi, v, getOperandOffset(outsOff, oi),
                        axes, order, p, parMask, prodPar, scope, store, finals);
      continue;
    }
    FailureOr<Value> result = emitValueOutResult(
        builder, loc, op, oi, v, slotPerOut[oi], prodPar, finals);
    if (failed(result))
      return failure();
    opResults[resultIdx++] = *result;
  }
  if (resultIdx != op.getNumResults())
    return op.emitOpError("value-outs lowering: result/out arity mismatch");
  return success();
}

static LogicalResult lowerValueOuts(HCGenericOp op, ArrayRef<IterAxis> axes) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  MLIRContext *ctx = op.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();

  FailureOr<ValueOutsLanes> lanes = collectValueOutsLanes(op, axes);
  if (failed(lanes))
    return failure();
  if (failed(validateValueOutsLaneCount(op, lanes->prodPar)))
    return failure();

  // Per-parLane slot maps. `outs` requires a bijection on
  // [0, prodPar); `ins` permits duplicates (gather pattern).
  ArrayAttr insOff = op.getInsOffsetsAttr();
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  FailureOr<SmallVector<SmallVector<int64_t>>> slotPerOut =
      resolveValueOutsSlots(op, store, outsOff, *lanes);
  if (failed(slotPerOut))
    return failure();
  FailureOr<SmallVector<SmallVector<int64_t>>> slotPerIn =
      resolveValueInsSlots(op, store, insOff, *lanes);
  if (failed(slotPerIn))
    return failure();

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
    p.push_back(static_cast<int>(lanes->bounds[i]));
  }

  llvm::StringMap<Value> scope = buildZeroIterScope(builder, loc, axes);
  seedAmbientScope(op, scope);
  SmallVector<SmallVector<Value>> insLanes =
      emitInsLoadsLaned(builder, op, axes, order, p, scope, store);
  if (failed(overrideInsLanesForValueIns(builder, loc, op, *slotPerIn,
                                         lanes->prodPar, insLanes)))
    return failure();

  SmallVector<bool, 4> parMask(axes.size(), true);
  SmallVector<SmallVector<Value>> outsInit(
      lanes->prodPar, SmallVector<Value>(op.getOuts().size()));
  if (failed(emitOutsInit(builder, loc, op, *slotPerOut, outsOff, axes, order,
                          p, parMask, lanes->prodPar, scope, store, outsInit)))
    return failure();

  SmallVector<SmallVector<Value>> finals(
      lanes->prodPar, SmallVector<Value>(op.getOuts().size()));
  if (failed(emitBodyClonesPerLane(builder, loc, op, axes, *lanes, insLanes,
                                   outsInit, scope, finals)))
    return failure();

  SmallVector<Value, 2> opResults(op.getNumResults());
  if (failed(emitOutsFinalize(builder, loc, op, *slotPerOut, outsOff, axes,
                              order, p, parMask, lanes->prodPar, scope, store,
                              finals, opResults)))
    return failure();
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
