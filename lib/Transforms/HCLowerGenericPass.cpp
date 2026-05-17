// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `-hc-lower-generic`: unroll-and-merge codegen for `hc.generic`.
// Picks axis order + per-axis unroll partition under
// ixsimpl-provable divisibility, scores by symbolic contig merge,
// emits scf.parallel(par) / scf.for(red) with one accumulator per
// (parLane, out) slot. Contig groups of width G ride vector
// load/store through extract/from_elements.

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

// Search tunables. Beyond `kAxisOrderCap` skip permutations (n!
// blowup). `kFactorChoices` is powers of two only.
constexpr int kUnrollBudget = 32;
constexpr int kAxisOrderCap = 4;
static constexpr std::array<int, 6> kFactorChoices = {1, 2, 4, 8, 16, 32};

// `boundExpr` absent on opaque `index` bound → divisibility probe
// conservatively rejects factor > 1.
struct IterAxis {
  size_t origIdx;
  StringRef name;
  IterKind kind;
  Value bound;
  std::optional<sym::ExprHandle> boundExpr;
};

using Partition = SmallVector<int, 4>;
using AxisOrder = SmallVector<size_t, 4>;

// `ixs_check` returns UNKNOWN on already-folded sentinels; peek tag
// first, fall through to interval check. Null → UNKNOWN.
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

// `factor | bound`? Opaque bound → false (conservative).
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

// `b - a == 1` symbolically. UNKNOWN → false: uncertain stays scalar.
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

// `s_a -> v_a` substitution: collapses offset to integer for the
// no-scf.parallel value-outs path. Returns `expr` unchanged on OOM.
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

// `s_a -> s_a + delta_a` substitution for per-lane offset emission.
// `delta_a == 0` skipped. Returns `expr` unchanged on OOM.
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

// Lane k → per-axis deltas. Rightmost in `order` varies fastest.
// Output indexed by declaration position, not `order`.
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

// Lanes `[start, start + size)` with all pairwise offset diffs == 1.
// `size > 1` is merge-eligible.
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

// All permutations for `n <= kAxisOrderCap`, identity-only beyond.
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

// All partitions p with `prod(p) <= budget`, p_a ∈ kFactorChoices.
// Trivial `(1,...,1)` included.
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

// `boundExpr` from `!hc.idx<expr>` SSA type; plain `index` → absent.
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

// Body clones per innermost iteration.
static int prodOf(ArrayRef<int> p) {
  int prod = 1;
  for (int v : p)
    prod *= v;
  return prod;
}

// Sub-product over axes of the given kind.
static int prodOfKind(ArrayRef<IterAxis> axes, ArrayRef<int> p, IterKind kind) {
  int prod = 1;
  for (auto [ax, pa] : llvm::zip(axes, p))
    if (ax.kind == kind)
      prod *= pa;
  return prod;
}

// Per-lane offsets for one operand. `includeAxis` picks the varied
// axes: ins use all, outs parallel-only (verifier guarantees outs
// offset reduction-sym-free).
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

// Merge score: sum of `(group_size - 1)` over every operand's
// contig groups. Trivial partition scores 0.
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
    // Outs init load + final store both benefit; weight x2.
    for (const ContigGroup &g : findContigGroups(store, offsets))
      score += 2 * (g.size - 1);
    (void)out;
  }
  return score;
}

// Highest merge score, divisibility-filtered. Ties resolve lex on
// (order, p). Falls back to trivial `(1,...,1)`.
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

// `!hc.idx<...>` → `index` via UCC; downstream reconcile folds it.
static Value castIdxToIndex(OpBuilder &builder, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), v)
      .getResult(0);
}

// Compile-time rank-1 lane count from a value carrier. Symbolic
// shapes rejected: compose needs an integer.
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

// Any non-ptr outs → forces fully-unrolled compose path.
static bool hasValueOuts(HCGenericOp op) {
  for (Value v : op.getOuts())
    if (!isa<PtrType>(v.getType()))
      return true;
  return false;
}

// Any non-ptr ins → forces fully-unrolled gather (extract per lane);
// partition path has no shape for value-typed ins.
static bool hasValueIns(HCGenericOp op) {
  for (Value v : op.getIns())
    if (!isa<PtrType>(v.getType()))
      return true;
  return false;
}

static std::string formatType(Type t) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  t.print(os);
  return buf;
}

// `!hc.ptr` with non-null element type (load/store needs pointee).
static bool isValidPtrOperand(Value v) {
  auto p = dyn_cast<PtrType>(v.getType());
  return p && p.getElementType();
}

// Diagnostic for malformed non-ptr operand types.
static std::string formatBadOperandTypeError(StringRef kind, size_t idx,
                                             Type ty) {
  return (Twine(kind) + " #" + Twine(idx) + " has type '" + formatType(ty) +
          "'; expected !hc.ptr<...> with element type or a rank-1 "
          "fixed-lane carrier with integer-literal shape")
      .str();
}

// Value-typed operand accepted iff carrier has compile-time rank-1
// lane count. Sets `hasValIn`/`hasValOut` if any non-ptr admitted.
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

// Offset arity / rank parity enforced upstream by op verifier +
// operand-type gate.
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

// Compile-time integer bound: `arith.constant` or
// `!hc.idx<"<int>">`-typed SSA.
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

// Reduction iter needs cross-lane carry the single-SSA-result
// boundary doesn't model.
static std::optional<std::string> diagnoseAllParallelIters(HCGenericOp op) {
  for (auto [i, k] : llvm::enumerate(op.getIterKindsAttr()))
    if (cast<IterKindAttr>(k).getValue() != IterKind::Parallel)
      return (Twine("iter #") + Twine(i) +
              " kind is reduction (value-typed operand needs all-parallel "
              "iters)")
          .str();
  return std::nullopt;
}

// Terminator must be `hc.yield` or `hc.yield_predicated`.
static std::optional<std::string> diagnoseGenericTerminator(HCGenericOp op) {
  Operation &term = op.getBody().front().back();
  if (!isa<HCYieldOp, HCYieldPredicatedOp>(&term))
    return (Twine("body terminator '") + term.getName().getStringRef() +
            "' is not hc.yield or hc.yield_predicated")
        .str();
  return std::nullopt;
}

// Value-typed operand offset must be iter-sym only: slot-eval
// constant-folds per lane. Ptr operands skip — `emitOffset`
// resolves ambient syms.
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

static bool isCollectiveCandidate(HCGenericOp op, ArrayRef<IterAxis> axes);

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
  // Terminator gate applies to every cloneBody site.
  if (auto r = diagnoseGenericTerminator(op))
    return r;
  // Remaining gates are value-outs-only; collective takes the
  // workgroup-tile path and bypasses them.
  if (isCollectiveCandidate(op, {}))
    return std::nullopt;
  if (auto r = diagnoseConstantIterBounds(op))
    return r;
  if (auto r = diagnoseAllParallelIters(op))
    return r;
  return diagnoseAmbientOffsetSyms(op);
}

// Single composed offset at `idx`. Rank-1 already checked.
static ExprAttr getOperandOffset(ArrayAttr arrayAttr, size_t idx) {
  return cast<ExprAttr>(cast<ArrayAttr>(arrayAttr[idx])[0]);
}

// Seed scope from `ambient_idxs` / `ambient_idx_syms`. Iter /
// partition writes later override on name collision.
static void seedAmbientScope(HCGenericOp op, llvm::StringMap<Value> &scope) {
  ArrayAttr syms = op.getAmbientIdxSymsAttr();
  OperandRange vals = op.getAmbientIdxs();
  for (auto [val, symAttr] :
       llvm::zip_equal(vals, syms.getAsRange<StringAttr>()))
    scope.try_emplace(symAttr.getValue(), val);
}

// Emit one offset as `index`-typed SSA via `hc.idx_apply`. Binds
// only free syms present in `loopScope`; unbound names stay free
// and resolve downstream via kernel-arg-bundle.
static Value emitOffset(OpBuilder &builder, Location loc, ExprAttr offsetExpr,
                        const llvm::StringMap<Value> &loopScope) {
  llvm::StringSet<> freeSyms;
  sym::walkSymbolNames(offsetExpr.getValue(),
                       [&](StringRef name) { freeSyms.insert(name); });

  // Lex-sort binding order: StringMap is unordered, IR text needs
  // determinism.
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

// Body apply ops may reference iter syms unbound; post-clone scan
// appends missing bindings. Expression unchanged.
namespace {
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

// In-scope iter syms missing from `existingSyms`. First-seen
// dedupe; caller sorts.
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

// Rebuild apply with `additions` appended to `symbols` / operands.
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

  // Lex-sort: deterministic IR text.
  llvm::sort(additions);
  Operation *replacement = createApplyOpWithIterSyms(
      builder, cloned, info->isIdx, additions, info->existingSyms, iterScope);
  for (auto [origRes, newRes] :
       llvm::zip_equal(nested.getResults(), replacement->getResults()))
    mapping.map(origRes, newRes);
  cloned->replaceAllUsesWith(replacement);
  cloned->erase();
}

// Coerce mask to `i1` / `vector<Nxi1>` for `arith.select`. UCC
// pair folds away when producer is already i1.
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

// Plain yield: raw vals, null masks. Predicated: vals + i1 masks.
// Predicated routes to ptr_store_pred — dodges OOB write race.
static LogicalResult
cloneBodyPredicatedYield(OpBuilder &builder, HCGenericOp op,
                         HCYieldPredicatedOp pyield, ValueRange outsVals,
                         const IRMapping &mapping,
                         SmallVectorImpl<Value> &yieldedOut,
                         SmallVectorImpl<Value> &yieldedMasksOut) {
  auto vals = pyield.getValues();
  auto masks = pyield.getMasks();
  if (vals.size() != outsVals.size())
    return op.emitOpError(
        "yield_predicated arity does not match outs operand count");
  if (vals.size() != masks.size())
    return op.emitOpError("yield_predicated values/masks size mismatch");
  yieldedOut.reserve(vals.size());
  yieldedMasksOut.reserve(vals.size());
  Location loc = op.getLoc();
  for (auto [v, m] : llvm::zip_equal(vals, masks)) {
    Value mapped = mapping.lookupOrDefault(v);
    Value mappedMask = mapping.lookupOrDefault(m);
    Value i1Mask = coerceMaskToI1(builder, loc, mappedMask, mapped.getType());
    yieldedOut.push_back(mapped);
    yieldedMasksOut.push_back(i1Mask);
  }
  return success();
}

// `yieldedMasksOut[i]` null on plain yield, i1 mask on predicated.
static LogicalResult cloneBody(OpBuilder &builder, HCGenericOp op,
                               ValueRange insVals, ValueRange outsVals,
                               const llvm::StringMap<Value> &iterScope,
                               SmallVectorImpl<Value> &yieldedOut,
                               SmallVectorImpl<Value> &yieldedMasksOut) {
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
  yieldedMasksOut.clear();
  if (auto yield = dyn_cast<HCYieldOp>(&term)) {
    yieldedOut.reserve(yield.getValues().size());
    yieldedMasksOut.assign(yield.getValues().size(), Value{});
    for (Value v : yield.getValues())
      yieldedOut.push_back(mapping.lookupOrDefault(v));
    return success();
  }
  if (auto pyield = dyn_cast<HCYieldPredicatedOp>(&term))
    return cloneBodyPredicatedYield(builder, op, pyield, outsVals, mapping,
                                    yieldedOut, yieldedMasksOut);
  return op.emitOpError(
      "body must end with `hc.yield` or `hc.yield_predicated`");
}

// Null mask: passthrough. Else: `arith.select(mask, raw, init)`.
static Value blendOrPassthrough(OpBuilder &builder, Location loc, Value raw,
                                Value mask, Value init) {
  if (!mask)
    return raw;
  return arith::SelectOp::create(builder, loc, mask, raw, init).getResult();
}

// One lane's offset as `index`-typed SSA: substitute
// `s_a -> s_a + delta_a` into the expression, then `emitOffset`.
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

// Project delta onto `subOrder` axes, fold back to a flat lane idx.
// Maps full lane to its parLane carry slot.
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

// Group load: scalar for size 1, `vector<GxT>` + extracts otherwise.
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

// Group store: scalar / `vector<GxT>` via `vector.from_elements`.
// Non-empty `laneMasks` routes through `hc.ptr_store_pred`. A group
// is uniformly masked or unmasked (mask is per-slot, not per-lane).
static void emitGroupStore(OpBuilder &builder, Location loc, Type elemTy,
                           Value baseAddr, const ContigGroup &g,
                           ArrayRef<Value> laneVals,
                           ArrayRef<Value> laneMasks) {
  bool masked = !laneMasks.empty() && laneMasks[g.start];
  if (g.size == 1) {
    if (masked) {
      HCPtrStorePredOp::create(builder, loc, laneVals[g.start], baseAddr,
                               laneMasks[g.start]);
    } else {
      HCPtrStoreOp::create(builder, loc, laneVals[g.start], baseAddr);
    }
    return;
  }
  SmallVector<Value> elems;
  elems.reserve(g.size);
  for (int k = 0; k < g.size; ++k)
    elems.push_back(laneVals[g.start + k]);
  auto vecTy = mlir::VectorType::get({g.size}, elemTy);
  Value vec =
      vector::FromElementsOp::create(builder, loc, vecTy, elems).getResult();
  if (!masked) {
    HCPtrStoreOp::create(builder, loc, vec, baseAddr);
    return;
  }
  SmallVector<Value> maskElems;
  maskElems.reserve(g.size);
  for (int k = 0; k < g.size; ++k)
    maskElems.push_back(laneMasks[g.start + k]);
  auto maskTy =
      mlir::VectorType::get({g.size}, IntegerType::get(elemTy.getContext(), 1));
  Value maskVec =
      vector::FromElementsOp::create(builder, loc, maskTy, maskElems)
          .getResult();
  HCPtrStorePredOp::create(builder, loc, vec, baseAddr, maskVec);
}

// Per-lane loads for ptr-typed ins, flat `[in_idx][lane]`. Value
// ins left empty here — only the unrolled path reaches them and
// fills via gather extracts.
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

// Outs init loads at parallel-iter scope. Flat layout
// `[par_lane * numOuts + out_idx]` for direct `iter_args` use.
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

// Outs stores at parallel-iter scope, contig-group merged. Empty
// `flatRaw`/`flatMasks` → carry path. Non-empty + predicated slot
// → `hc.ptr_store_pred` (skips OOB read-modify-write race).
static void emitOutsStoresPartitioned(OpBuilder &builder, HCGenericOp op,
                                      ArrayRef<IterAxis> axes,
                                      ArrayRef<size_t> order, ArrayRef<int> p,
                                      ValueRange flatFinals, ValueRange flatRaw,
                                      ValueRange flatMasks,
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
    // Predicated slot uses raw+mask via ptr_store_pred; mask
    // uniformity is per-slot, so any-parLane non-null implies all.
    bool slotMasked = !flatMasks.empty() && flatMasks[0 * numOuts + oi];
    SmallVector<Value> laneVals(prodPar);
    SmallVector<Value> laneMasks;
    for (int pl = 0; pl < prodPar; ++pl)
      laneVals[pl] = slotMasked ? flatRaw[pl * numOuts + oi]
                                : flatFinals[pl * numOuts + oi];
    if (slotMasked) {
      laneMasks.resize(prodPar);
      for (int pl = 0; pl < prodPar; ++pl)
        laneMasks[pl] = flatMasks[pl * numOuts + oi];
    }
    for (const ContigGroup &g : groups) {
      Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes,
                                 order, p, parMask, g.start);
      Value addr = HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off)
                       .getResult();
      emitGroupStore(builder, loc, elemTy, addr, g, laneVals, laneMasks);
    }
  }
}

// scope binds iter syms to `iv_a + delta_a` for body applies.
// acc: blended carry. lastRaw/lastMask: unblended pair for
// pure-parallel ptr_store_pred routing.
static LogicalResult
emitOneLaneBody(OpBuilder &builder, HCGenericOp op, int lane,
                ArrayRef<IterAxis> axes, ArrayRef<size_t> order,
                ArrayRef<size_t> parOrder, ArrayRef<int> p,
                ArrayRef<SmallVector<Value>> insLanes, size_t numOuts,
                const llvm::StringMap<Value> &scope,
                MutableArrayRef<SmallVector<Value>> acc,
                MutableArrayRef<SmallVector<Value>> lastRaw,
                MutableArrayRef<SmallVector<Value>> lastMask) {
  SmallVector<int, 4> delta = decomposeLane(lane, order, p);
  int parLane = computeSubLane(delta, parOrder, p);
  SmallVector<Value> insVals(insLanes.size());
  for (size_t ii = 0; ii < insLanes.size(); ++ii)
    insVals[ii] = insLanes[ii][lane];
  SmallVector<Value> outsVals = acc[parLane];
  SmallVector<Value> yielded;
  SmallVector<Value> masks;
  llvm::StringMap<Value> laneIterScope = scope;
  Location loc = op.getLoc();
  for (auto [ax, d] : llvm::zip_equal(axes, delta)) {
    Value base = scope.lookup(ax.name);
    if (!base)
      continue;
    if (d == 0) {
      laneIterScope[ax.name] = base;
      continue;
    }
    Value deltaConst =
        arith::ConstantIndexOp::create(builder, loc, d).getResult();
    laneIterScope[ax.name] =
        arith::AddIOp::create(builder, loc, base, deltaConst).getResult();
  }
  if (failed(cloneBody(builder, op, insVals, outsVals, laneIterScope, yielded,
                       masks)))
    return failure();
  if (yielded.size() != numOuts)
    return op.emitOpError("body yielded wrong arity");
  for (size_t oi = 0; oi < numOuts; ++oi) {
    acc[parLane][oi] =
        blendOrPassthrough(builder, loc, yielded[oi], masks[oi], outsVals[oi]);
    lastRaw[parLane][oi] = yielded[oi];
    lastMask[parLane][oi] = masks[oi];
  }
  return success();
}

// `blended`: carry form for scf.for. `raw`/`mask`: unblended pair
// for pure-parallel ptr_store_pred path.
struct PartitionFlat {
  SmallVector<Value> blended;
  SmallVector<Value> raw;
  SmallVector<Value> mask;
};

// Innermost emission: ins loads, prod(p) body clones in
// rightmost-fastest order. Reduction axes compose into the same
// accumulator slot per parLane.
static FailureOr<PartitionFlat>
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

  SmallVector<SmallVector<Value>> lastRaw(prodPar, SmallVector<Value>(numOuts));
  SmallVector<SmallVector<Value>> lastMask(prodPar,
                                           SmallVector<Value>(numOuts));
  for (int lane = 0; lane < prodAll; ++lane)
    if (failed(emitOneLaneBody(builder, op, lane, axes, order, parOrder, p,
                               insLanes, numOuts, scope, acc, lastRaw,
                               lastMask)))
      return failure();

  PartitionFlat out;
  out.blended.reserve(prodPar * numOuts);
  out.raw.reserve(prodPar * numOuts);
  out.mask.reserve(prodPar * numOuts);
  for (int pl = 0; pl < prodPar; ++pl)
    for (size_t oi = 0; oi < numOuts; ++oi) {
      out.blended.push_back(acc[pl][oi]);
      out.raw.push_back(lastRaw[pl][oi]);
      out.mask.push_back(lastMask[pl][oi]);
    }
  return out;
}

// Reduction scf.for nest stepping `p[a]`, threading the flat
// accumulator. Carries blended form only; raw/mask irrelevant here.
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
    if (depth == redOrder.size()) {
      auto flat = emitInnerBodyClones(builder, op, axes, order, p, iterArgs,
                                      localScope, store);
      if (failed(flat))
        return failure();
      return std::move(flat->blended);
    }
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

// One parallel-iter call: outs init, body or reduction nest, store.
// Pure-parallel predicated → ptr_store_pred (OOB-safe). Reduction
// stores blended unconditionally.
static LogicalResult lowerPartitionPerParallelBody(
    OpBuilder &builder, HCGenericOp op, ArrayRef<IterAxis> axes,
    ArrayRef<size_t> order, ArrayRef<int> p, ArrayRef<size_t> redOrder,
    llvm::StringMap<Value> &scope, sym::Store &store) {
  SmallVector<Value> initOuts =
      emitOutsInitLoadsPartitioned(builder, op, axes, order, p, scope, store);
  SmallVector<Value> finals;
  SmallVector<Value> finalsRaw;
  SmallVector<Value> finalsMask;
  if (redOrder.empty()) {
    auto flat = emitInnerBodyClones(builder, op, axes, order, p, initOuts,
                                    scope, store);
    if (failed(flat))
      return failure();
    finals = std::move(flat->blended);
    finalsRaw = std::move(flat->raw);
    finalsMask = std::move(flat->mask);
  } else {
    auto reduced = emitReductionNestPartitioned(
        builder, op, axes, order, p, redOrder, initOuts, scope, store);
    if (failed(reduced))
      return failure();
    finals = std::move(*reduced);
  }
  emitOutsStoresPartitioned(builder, op, axes, order, p, finals, finalsRaw,
                            finalsMask, scope, store);
  return success();
}

// ParallelOp bounds: [0, dim) step p[a], in `parIdx` order.
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

// Top-level emit: outer scf.parallel(par) step p[a], inner
// scf.for(red) nest, body cloned prod(p) per innermost iter.
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

// Element conversion must match the upstream UCC producer so
// canonicalize folds the `ptr<wg> ↔ bare_tensor` pair.
static Type convertBareTensorElement(Type t) {
  if (isa<PredType>(t))
    return IntegerType::get(t.getContext(), 1);
  if (t.isIntOrIndexOrFloat())
    return t;
  return {};
}

// `!hc.ptr<workgroup, T>` matching the upstream-planted UCC so
// canonicalize collapses the pair.
static PtrType workgroupPtrFor(BareTensorType bt) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(bt);
  Type elem = convertBareTensorElement(shaped.getSymbolicElementType());
  if (!elem)
    return PtrType();
  return PtrType::get(bt.getContext(), AddrSpace::Workgroup, elem);
}

// Workgroup-staged outs inside `gpu.launch` → collective path
// (shared tile needs cross-wave partition). Reduction iter ok:
// per-thread scf.for over its LDS slot.
static bool isCollectiveCandidate(HCGenericOp op, ArrayRef<IterAxis> /*axes*/) {
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
  return op->getParentOfType<gpu::LaunchOp>() != nullptr;
}

// Pin every outs to workgroup ptr. bare_tensor → UCC to backing
// ptr (folds with upstream pair at canonicalize).
static SmallVector<Value> collectiveOutsPtrs(OpBuilder &builder, Location loc,
                                             HCGenericOp op) {
  SmallVector<Value> outsPtrs(op.getOuts().size());
  for (auto [i, out] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(out.getType())) {
      outsPtrs[i] = out;
      continue;
    }
    auto bt = dyn_cast<BareTensorType>(out.getType());
    assert(bt && "isCollectiveCandidate accepted non-bare_tensor outs");
    PtrType ptrTy = workgroupPtrFor(bt);
    assert(ptrTy && "isCollectiveCandidate accepted unconvertible outs");
    outsPtrs[i] = UnrealizedConversionCastOp::create(builder, loc, ptrTy, out)
                      .getResult(0);
  }
  return outsPtrs;
}

// `total = prod(bounds)`, `chunks = ceil(total / wgSize)`.
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

// Unlinearise to per-axis coords, rightmost-fastest (flatten
// convention).
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

// Load + UCC-bridge to body arg type (e.g. `!hc.pred` ↔ `i1` LDS).
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

// Store + symmetric UCC bridge. Non-null `mask` →
// `hc.ptr_store_pred`.
static void storeCollectiveYielded(OpBuilder &builder, Location loc, Value ptr,
                                   Value off, Value yielded, Value mask) {
  Type elemTy = cast<PtrType>(ptr.getType()).getElementType();
  Value addr =
      HCPtrOffsetOp::create(builder, loc, ptr.getType(), ptr, off).getResult();
  Value toStore = yielded;
  if (toStore.getType() != elemTy)
    toStore = UnrealizedConversionCastOp::create(builder, loc, elemTy, toStore)
                  .getResult(0);
  if (mask)
    HCPtrStorePredOp::create(builder, loc, toStore, addr, mask);
  else
    HCPtrStoreOp::create(builder, loc, toStore, addr);
}

// Load every ins/outs at composed offset; result vectors parallel
// to op's ins/outs.
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

// Body clone + outs store for one in-range chunk-thread. Predicated
// slots → `hc.ptr_store_pred`, plain → unconditional.
static LogicalResult
emitCollectiveChunkInRange(OpBuilder &builder, Location loc, HCGenericOp op,
                           ArrayRef<Value> outsPtrs, ValueRange insVals,
                           ValueRange outsVals,
                           const llvm::StringMap<Value> &scope) {
  SmallVector<Value> yielded;
  SmallVector<Value> masks;
  if (failed(cloneBody(builder, op, insVals, outsVals, scope, yielded, masks)))
    return failure();
  if (yielded.size() != op.getOuts().size())
    return op.emitOpError("body yielded wrong arity");
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
    Value off = emitOffset(builder, loc, getOperandOffset(outsOff, oi), scope);
    storeCollectiveYielded(builder, loc, outsPtrs[oi], off, yielded[oi],
                           masks[oi]);
  }
  return success();
}

namespace {
// Per-kind axis metadata cached for the collective sweep.
struct CollectiveIterSplit {
  SmallVector<size_t> parIdx;
  SmallVector<size_t> redIdx;
  SmallVector<Value> parBounds;
  SmallVector<Value> redBounds;
};
} // namespace

static CollectiveIterSplit splitCollectiveIters(HCGenericOp op,
                                                ArrayRef<IterAxis> axes) {
  CollectiveIterSplit out;
  ValueRange bounds = op.getIterBounds();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i].kind == IterKind::Parallel) {
      out.parIdx.push_back(i);
      out.parBounds.push_back(bounds[i]);
    } else {
      out.redIdx.push_back(i);
      out.redBounds.push_back(bounds[i]);
    }
  }
  return out;
}

// Plain yield: passthrough. Predicated: select(mask, raw, carry).
static SmallVector<Value> blendYieldedAgainstCarry(OpBuilder &builder,
                                                   Location loc,
                                                   ArrayRef<Value> yielded,
                                                   ArrayRef<Value> masks,
                                                   ValueRange carries) {
  SmallVector<Value> blended(yielded.size());
  for (size_t oi = 0; oi < yielded.size(); ++oi)
    blended[oi] =
        blendOrPassthrough(builder, loc, yielded[oi], masks[oi], carries[oi]);
  return blended;
}

// Reduction iter syms bind to loop IVs through the nest; restore on
// unwind so siblings see no temp binding.
static LogicalResult emitCollectiveChunkReductionNest(
    OpBuilder &builder, Location loc, HCGenericOp op, ArrayRef<IterAxis> axes,
    const CollectiveIterSplit &split, ArrayRef<Value> outsPtrs,
    const llvm::StringMap<Value> &parScope) {
  ArrayAttr outsOff = op.getOutsOffsetsAttr();
  ArrayAttr insOff = op.getInsOffsetsAttr();
  Block &body = op.getBody().front();
  size_t numIns = op.getIns().size();

  SmallVector<Value> outsInit(op.getOuts().size());
  for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
    Value off =
        emitOffset(builder, loc, getOperandOffset(outsOff, oi), parScope);
    Type bodyArgTy = body.getArgument(numIns + oi).getType();
    outsInit[oi] = loadCollectiveOperandElement(builder, loc, outsPtrs[oi], off,
                                                bodyArgTy);
  }

  std::function<FailureOr<SmallVector<Value>>(size_t, ValueRange,
                                              llvm::StringMap<Value> &)>
      build = [&](size_t depth, ValueRange carries,
                  llvm::StringMap<Value> &localScope)
      -> FailureOr<SmallVector<Value>> {
    if (depth == split.redIdx.size()) {
      SmallVector<Value> insVals(numIns);
      for (size_t ii = 0; ii < numIns; ++ii) {
        Value off =
            emitOffset(builder, loc, getOperandOffset(insOff, ii), localScope);
        Type bodyArgTy = body.getArgument(ii).getType();
        insVals[ii] = loadCollectiveOperandElement(
            builder, loc, op.getIns()[ii], off, bodyArgTy);
      }
      SmallVector<Value> outsVals(carries.begin(), carries.end());
      SmallVector<Value> yielded;
      SmallVector<Value> masks;
      if (failed(cloneBody(builder, op, insVals, outsVals, localScope, yielded,
                           masks)))
        return failure();
      if (yielded.size() != op.getOuts().size())
        return op.emitOpError("body yielded wrong arity");
      return blendYieldedAgainstCarry(builder, loc, yielded, masks, carries);
    }
    size_t ri = split.redIdx[depth];
    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    Value ub = castIdxToIndex(builder, loc, split.redBounds[depth]);
    Value step = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
    auto forOp = scf::ForOp::create(builder, loc, c0, ub, step, carries);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    StringRef name = axes[ri].name;
    Value saved = localScope.lookup(name);
    localScope[name] = forOp.getInductionVar();
    auto inner = build(depth + 1, forOp.getRegionIterArgs(), localScope);
    if (saved)
      localScope[name] = saved;
    else
      localScope.erase(name);
    if (failed(inner))
      return failure();
    scf::YieldOp::create(builder, loc, *inner);
    return SmallVector<Value>(forOp.getResults().begin(),
                              forOp.getResults().end());
  };

  llvm::StringMap<Value> nestScope = parScope;
  auto finals = build(0, ValueRange(outsInit), nestScope);
  if (failed(finals))
    return failure();
  // Carry already folded masking via blendOrPassthrough; store
  // unconditionally. All-false reduction mask writes init back —
  // OOB race ptr_store_pred dodges in the pure-parallel path.
  for (size_t oi = 0; oi < op.getOuts().size(); ++oi) {
    Value off =
        emitOffset(builder, loc, getOperandOffset(outsOff, oi), parScope);
    storeCollectiveYielded(builder, loc, outsPtrs[oi], off, (*finals)[oi],
                           /*mask=*/Value{});
  }
  return success();
}

// One chunk-thread body inside `lin < total` guard. All-parallel:
// load, body, store. Mixed: per-thread scf.for nest over reductions
// with carry in registers, one LDS store per slot at end.
static LogicalResult emitCollectiveChunkBody(
    OpBuilder &builder, Location loc, HCGenericOp op, ArrayRef<IterAxis> axes,
    const CollectiveIterSplit &split, Value lin, ArrayRef<Value> outsPtrs) {
  SmallVector<Value> parCoords = unlinearizeCollectiveCoords(
      builder, loc, lin, ValueRange(split.parBounds));
  llvm::StringMap<Value> scope;
  seedAmbientScope(op, scope);
  for (auto [pi, c] : llvm::zip(split.parIdx, parCoords))
    scope[axes[pi].name] = c;

  if (split.redIdx.empty()) {
    SmallVector<Value> insVals, outsVals;
    loadCollectiveOperands(builder, loc, op, outsPtrs, scope, insVals,
                           outsVals);
    return emitCollectiveChunkInRange(builder, loc, op, outsPtrs, insVals,
                                      outsVals, scope);
  }
  return emitCollectiveChunkReductionNest(builder, loc, op, axes, split,
                                          outsPtrs, scope);
}

// Wave-strided dispatch: each thread runs `c * wgSize + lin_tid`,
// trailing partial chunk gated by `lin < total`. Cross-generic LDS
// sync owned by `hc-insert-workgroup-barriers` upstream — don't
// emit `gpu.barrier` here (canonicalizer won't drop duplicates).
static LogicalResult lowerCollective(HCGenericOp op, ArrayRef<IterAxis> axes) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  auto tidAndSize = linearizedThreadAndSize(builder, loc, op);
  if (failed(tidAndSize))
    return op.emitOpError("collective dispatch requires a gpu.launch parent");
  auto [linTid, wgSize] = *tidAndSize;

  CollectiveIterSplit split = splitCollectiveIters(op, axes);
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  auto [total, chunks] = collectiveTotalAndChunks(
      builder, loc, ValueRange(split.parBounds), wgSize);
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
        emitCollectiveChunkBody(builder, loc, op, axes, split, lin, outsPtrs);
  }
  if (failed(bodyStatus))
    return failure();

  // RAUW result to its outs SSA so downstream UCCs see the input
  // chain and fold cleanly.
  for (auto [res, out] : llvm::zip(op.getResults(), op.getOuts()))
    res.replaceAllUsesWith(out);
  return success();
}

// Constant int bound. Gate already pinned one of the two forms.
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

// Per-axis decompose against int bounds, rightmost-fastest.
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

// Seed scope `iter_sym -> 0`: substituteIterDeltas shifts each by
// its lane delta, leaving the residual sym which emitOffset must
// list as an explicit (constant) operand.
static llvm::StringMap<Value>
buildZeroIterScope(OpBuilder &builder, Location loc, ArrayRef<IterAxis> axes) {
  Value cZero = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  llvm::StringMap<Value> scope;
  for (const IterAxis &ax : axes)
    scope[ax.name] = cZero;
  return scope;
}

// Builtin element compatible with `VectorType` (extract/compose).
// Null otherwise.
static Type asBuiltinElementType(Type t) {
  if (isa<PredType>(t))
    return IntegerType::get(t.getContext(), 1);
  if (t.isIntOrIndexOrFloat())
    return t;
  return Type();
}

// Per-lane integer slot via iter-sym constant substitution.
static std::optional<int64_t> resolveOperandSlot(sym::Store &store,
                                                 ExprAttr origOff,
                                                 ArrayRef<StringRef> iterNames,
                                                 ArrayRef<int> vals) {
  sym::ExprHandle subbed =
      substituteIterValues(store, origOff.getValue(), iterNames, vals);
  return sym::getIntegerLiteralValue(subbed);
}

// Fully-unrolled path for any value-typed operand.
// Outs offset: bijection on `[0, prodPar)`.
// Ins offset: integer in `[0, count)` per lane; gather, repeats ok.
namespace {
struct ValueOutsLanes {
  SmallVector<int64_t> bounds;
  SmallVector<StringRef> iterNames;
  int64_t prodPar = 1;
};

// `vector<NxBuiltin>` carrier value operands UCC through.
// `elemTy` symbolic, `builtinElem` surrogate, `vecTy` the bare vec.
struct VectorCarrier {
  Type elemTy;
  Type builtinElem;
  mlir::VectorType vecTy;
};
} // namespace

// Per-axis bounds, names, total lane count. Fails on non-constant.
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

// Value outs must have rank-1 lane count == `prodPar`.
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

// Per-lane slot via constant-eval. `requireBijection` rejects gaps
// + dupes; else gather (dupes allowed).
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

// Outs bijection on `[0, prodPar)`.
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

// `vector<NxBuiltin>` carrier shape. Fails if element has no
// builtin surrogate.
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

// UCC `v` to `vecTy`. Same shape fast-paths. Sees only value
// carriers — LDS-backed bare_tensor already swapped to ptr upstream.
static Value castToVectorCarrier(OpBuilder &builder, Location loc, Value v,
                                 mlir::VectorType vecTy) {
  if (v.getType() == Type(vecTy))
    return v;
  return UnrealizedConversionCastOp::create(builder, loc, vecTy, v)
      .getResult(0);
}

// `vector.extract` + UCC back to body element type.
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

// Value ins gather: UCC to vector carrier, extract per lane at the
// constant-eval'd slot, UCC back to body element.
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

// Ptr outs init: per-lane load with contig-group merging.
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

// Value out init: per-lane extract at the eval'd slot.
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

// Per-parLane outs init across value (extract) and ptr (load) outs.
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

// Full unroll: each iter sym a per-lane constant. Start from
// ambient scope so applies referencing $WG / kernel-arg /
// structured-loop syms get bindings; per-lane constants stamp over.
//
// finals: consumer form (blended). finalsRaw/finalsMask: unblended
// pair for ptr_store_pred.
static LogicalResult emitBodyClonesPerLane(
    OpBuilder &builder, Location loc, HCGenericOp op, ArrayRef<IterAxis> axes,
    const ValueOutsLanes &lanes, ArrayRef<SmallVector<Value>> insLanes,
    ArrayRef<SmallVector<Value>> outsInit, const llvm::StringMap<Value> &scope,
    SmallVectorImpl<SmallVector<Value>> &finals,
    SmallVectorImpl<SmallVector<Value>> &finalsRaw,
    SmallVectorImpl<SmallVector<Value>> &finalsMask) {
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
    SmallVector<Value> masks;
    if (failed(cloneBody(builder, op, insVals, outsInit[lane], laneIterScope,
                         yielded, masks)))
      return failure();
    if (yielded.size() != numOuts)
      return op.emitOpError("body yielded wrong arity");
    finalsRaw[lane] = yielded;
    finalsMask[lane] = masks;
    finals[lane].resize(numOuts);
    for (size_t oi = 0; oi < numOuts; ++oi)
      finals[lane][oi] = blendOrPassthrough(builder, loc, yielded[oi],
                                            masks[oi], outsInit[lane][oi]);
  }
  return success();
}

// Ptr outs stores, contig-group merged. Predicated slot →
// `hc.ptr_store_pred` (skips OOB write).
static void emitPtrOutsStores(OpBuilder &builder, Location loc, HCGenericOp op,
                              size_t oi, Value v, ExprAttr origOff,
                              ArrayRef<IterAxis> axes, ArrayRef<size_t> order,
                              ArrayRef<int> p, ArrayRef<bool> parMask,
                              int64_t prodPar,
                              const llvm::StringMap<Value> &scope,
                              sym::Store &store,
                              ArrayRef<SmallVector<Value>> finalsRaw,
                              ArrayRef<SmallVector<Value>> finalsMask) {
  Type elemTy = cast<PtrType>(v.getType()).getElementType();
  SmallVector<sym::ExprHandle> offs =
      laneOffsets(store, origOff, axes, order, p, parMask, prodPar);
  auto groups = findContigGroups(store, offs);
  bool slotMasked = prodPar > 0 && finalsMask[0][oi];
  SmallVector<Value> laneVals(prodPar);
  SmallVector<Value> laneMasks;
  for (int64_t parLane = 0; parLane < prodPar; ++parLane)
    laneVals[parLane] = finalsRaw[parLane][oi];
  if (slotMasked) {
    laneMasks.resize(prodPar);
    for (int64_t parLane = 0; parLane < prodPar; ++parLane)
      laneMasks[parLane] = finalsMask[parLane][oi];
  }
  for (const ContigGroup &g : groups) {
    Value off = emitLaneOffset(builder, loc, origOff, scope, store, axes, order,
                               p, parMask, g.start);
    Value addr =
        HCPtrOffsetOp::create(builder, loc, v.getType(), v, off).getResult();
    emitGroupStore(builder, loc, elemTy, addr, g, laneVals, laneMasks);
  }
}

// Value out compose: UCC each lane to builtin element,
// `vector.from_elements` at the slot map, UCC back to bare carrier.
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

// Per-out: ptr store group, or value compose into `opResults`.
static LogicalResult
emitOutsFinalize(OpBuilder &builder, Location loc, HCGenericOp op,
                 ArrayRef<SmallVector<int64_t>> slotPerOut, ArrayAttr outsOff,
                 ArrayRef<IterAxis> axes, ArrayRef<size_t> order,
                 ArrayRef<int> p, ArrayRef<bool> parMask, int64_t prodPar,
                 const llvm::StringMap<Value> &scope, sym::Store &store,
                 ArrayRef<SmallVector<Value>> finals,
                 ArrayRef<SmallVector<Value>> finalsRaw,
                 ArrayRef<SmallVector<Value>> finalsMask,
                 SmallVectorImpl<Value> &opResults) {
  size_t resultIdx = 0;
  for (auto [oi, v] : llvm::enumerate(op.getOuts())) {
    if (isa<PtrType>(v.getType())) {
      emitPtrOutsStores(builder, loc, op, oi, v, getOperandOffset(outsOff, oi),
                        axes, order, p, parMask, prodPar, scope, store,
                        finalsRaw, finalsMask);
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

  // Reuse partition emitter with full unroll: p[a] = bounds[a],
  // declaration order. Lets contig analysis collapse 16x16
  // unit-stride into one `vector<256xT>` before LLVM vectorizer.
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
  SmallVector<SmallVector<Value>> finalsRaw(
      lanes->prodPar, SmallVector<Value>(op.getOuts().size()));
  SmallVector<SmallVector<Value>> finalsMask(
      lanes->prodPar, SmallVector<Value>(op.getOuts().size()));
  if (failed(emitBodyClonesPerLane(builder, loc, op, axes, *lanes, insLanes,
                                   outsInit, scope, finals, finalsRaw,
                                   finalsMask)))
    return failure();

  SmallVector<Value, 2> opResults(op.getNumResults());
  if (failed(emitOutsFinalize(builder, loc, op, *slotPerOut, outsOff, axes,
                              order, p, parMask, lanes->prodPar, scope, store,
                              finals, finalsRaw, finalsMask, opResults)))
    return failure();
  op->replaceAllUsesWith(opResults);
  return success();
}

// Dispatch precedence: collective (workgroup-shared outs in
// `gpu.launch`) → value-typed operands → partition (all-ptr).
// Order matters: LDS-backed bare_tensor must partition across the
// wave even though carrier type would admit the value path.
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

  // Partition path is all-ptr-outs → no SSA results. Guard catches
  // future gate relaxations that would drop results silently.
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

    // Post-walk: surviving `hc.generic` is fatal. Re-run gate so
    // future paths that succeed-without-erase also surface here.
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
