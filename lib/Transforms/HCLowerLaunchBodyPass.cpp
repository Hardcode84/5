// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-launch-body`, the launch-body lowering pass that runs
// after HC kernels have been wrapped in `gpu.launch`.
//
// Workgroup-AS storage and kernel-arg buffers are both off memref. Workgroup
// tiles materialize as `hc.alloc` + `hc.ptr_*` against `!hc.ptr<workgroup, T>`;
// kernel-arg buffers arrive as a `(!hc.ptr<global, T>, dim*, stride*)` UCC
// fragment that this pass walks to recover dims/strides and to emit
// `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`. The
// downstream `hc-lower-to-llvm` finishes the LLVM-dialect lowering.
// `doc/layouts.md` "hc.ptr and memory ops" holds the contract.

#include "hc/Transforms/Passes.h"

#include "LaunchUtils.h"
#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

#include <type_traits>

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERLAUNCHBODY
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

static ixs_node *rawNode(ExprAttr expr) {
  return const_cast<ixs_node *>(expr.getNode());
}

static ixs_node *rawNode(PredAttr pred) {
  return const_cast<ixs_node *>(pred.getNode());
}

static std::optional<StringRef> exactSymbolName(ExprAttr expr) {
  if (!expr)
    return std::nullopt;
  ixs_node *node = rawNode(expr);
  if (ixs_node_tag(node) != IXS_SYM)
    return std::nullopt;
  return StringRef(ixs_node_sym_name(node));
}

static std::optional<StringRef> exactSymbolName(Type type) {
  auto idx = dyn_cast<IdxType>(type);
  return idx ? exactSymbolName(idx.getExpr()) : std::nullopt;
}

static bool isOneNode(ixs_node *node) {
  return ixs_node_tag(node) == IXS_INT && ixs_node_int_val(node) == 1;
}

static Type convertElementType(Type type) {
  if (isa<PredType>(type))
    return IntegerType::get(type.getContext(), 1);
  if (type.isIntOrIndexOrFloat())
    return type;
  return {};
}

static FailureOr<SmallVector<int64_t>> staticIntegerShape(ShapeAttr shape,
                                                          Operation *diagOp) {
  SmallVector<int64_t> dims;
  dims.reserve(shape.getDims().size());
  for (auto [index, attr] : llvm::enumerate(shape.getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<int64_t> value =
        expr ? sym::getIntegerLiteralValue(sym::ExprHandle(expr.getNode()))
             : std::nullopt;
    if (!value || *value < 0) {
      if (diagOp)
        return diagOp->emitOpError("expected static non-negative integer "
                                   "dimension at #")
               << index << ", got " << attr;
      return failure();
    }
    dims.push_back(*value);
  }
  return dims;
}

static FailureOr<SmallVector<int64_t>>
staticIntegerShape(SymbolicallyShapedTypeInterface shaped) {
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(shaped.getSymbolicShape(), nullptr);
  if (failed(dims))
    return failure();
  return *dims;
}

// Bare tensors collapse to `!hc.ptr<workgroup, T>` — the launch-body owns
// the workgroup tile only as opaque-flat storage; the rank and layout were
// frontend semantics, and once memory is in hand the lowering treats every
// LDS allocation as a flat element-typed buffer. Static dims are required
// because `hc.alloc` workgroup needs a constant element count downstream.
static Type convertBareTensorType(BareTensorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  if (failed(staticIntegerShape(shaped)))
    return {};
  return PtrType::get(type.getContext(), AddrSpace::Workgroup, element);
}

// Total element count for the workgroup tile — product of static dims.
static FailureOr<int64_t> bareTensorElementCount(BareTensorType type) {
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(type));
  if (failed(dims))
    return failure();
  int64_t total = 1;
  for (int64_t d : *dims) {
    if (d < 0)
      return failure();
    total *= d;
  }
  return total;
}

static Type convertBareVectorType(BareVectorType type) {
  auto shaped = cast<SymbolicallyShapedTypeInterface>(type);
  Type element = convertElementType(shaped.getSymbolicElementType());
  if (!element)
    return {};
  FailureOr<SmallVector<int64_t>> dims = staticIntegerShape(shaped);
  if (failed(dims))
    return {};
  if (dims->empty())
    return element;
  for (int64_t dim : *dims)
    if (dim <= 0)
      return {};
  return mlir::VectorType::get(*dims, element);
}

static Value resolveToBundleIndex(Value value);

static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                             Location loc) {
  if (inputs.size() != 1)
    return {};
  // Conversion-driver materialization (`idx<sym>` → `index`, etc.):
  // when the source is a post-flatten retype output that chains back
  // to a kernel-arg bundle, prefer the bundle's `index`-typed input
  // value over a fresh UCC. See `resolveToBundleIndex` for why we
  // can't leave this for `reconcileUnrealizedCasts` to clean up.
  if (type.isIndex()) {
    if (Value direct = resolveToBundleIndex(inputs.front()))
      return direct;
  }
  return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
      .getResult(0);
}

static Value castIfNeeded(OpBuilder &builder, Location loc, Value value,
                          Type type) {
  if (value.getType() == type)
    return value;
  return UnrealizedConversionCastOp::create(builder, loc, type, value)
      .getResult(0);
}

class HCLaunchBodyTypeConverter : public TypeConverter {
public:
  HCLaunchBodyTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });
    addConversion(
        [](IdxType type) -> Type { return IndexType::get(type.getContext()); });
    addConversion([](PredType type) -> Type {
      return IntegerType::get(type.getContext(), 1);
    });
    addConversion([](BareTensorType type) -> std::optional<Type> {
      if (Type converted = convertBareTensorType(type))
        return converted;
      return Type(type);
    });
    addConversion([](BareVectorType type) -> std::optional<Type> {
      if (Type converted = convertBareVectorType(type))
        return converted;
      return Type(type);
    });
    addConversion([&](TupleType type) -> std::optional<Type> {
      SmallVector<Type> elements;
      if (failed(convertTypes(type.getTypes(), elements)))
        return std::nullopt;
      return TupleType::get(type.getContext(), elements);
    });
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

struct BoundValues {
  llvm::StringMap<Value> symbols;

  void bind(StringRef name, Value value) {
    if (!name.empty())
      symbols.try_emplace(name, value);
  }

  Value lookup(StringRef name) const {
    auto it = symbols.find(name);
    return it == symbols.end() ? Value{} : it->second;
  }
};

// Kernel-arg fragment: the UCC at the launch boundary expands a single
// `!hc.buffer<T, [dims]>` into 1 + 2N values — a global pointer, then per-axis
// dim values, then per-axis element-stride values, in that order. The host
// wrapper builds it (`hc-lower-kernels-to-gpu-launch`); launch-body sees the
// UCC, walks back through it, and turns the per-element load/store into
// `hc.ptr_offset` + `hc.ptr_load[_pred]` / `hc.ptr_store[_pred]`.
struct KernelArgSource {
  Value ptr;
  SmallVector<Value> dims;
  SmallVector<Value> strides;

  unsigned rank() const { return dims.size(); }
};

static std::optional<KernelArgSource> resolveKernelArg(Value source);

// Pure query: walks defining ops to find the underlying kernel-arg
// `(ptr, dims..., strides...)` UCC. Handles both the pre-flatten
// direct kernel-arg shape (one UCC carrying the bundle) and the
// post-flatten chain (a multi-output UCC retypes the bundle to a
// rank-1 carrier plus idx-typed aux). For the post-flatten chain we
// return the inner rank-N source — the consumer decides whether to
// use it directly (e.g. `AdaptGenericOp` just needs `ptr`) or to
// collapse to a rank-1 view (`flatKernelArgView`, IR-mutating).
// `cast` is the multi-output bare-carrier UCC that the post-flatten
// pipeline plants over a rank-1 `BufferType` view. Caller has
// established `cast` is a UCC; we just check the shape and recurse
// into the original kernel-arg input.
static std::optional<KernelArgSource>
resolveKernelArgViaBareCarrier(UnrealizedConversionCastOp cast, Value source) {
  if (cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1 ||
      source != cast.getOutputs()[0])
    return std::nullopt;
  auto bufOut = dyn_cast<BufferType>(source.getType());
  if (!bufOut || bufOut.getShape().getDims().size() != 1)
    return std::nullopt;
  return resolveKernelArg(cast.getInputs()[0]);
}

// Validate the canonical kernel-arg UCC shape: 1 output, and inputs
// of the form `ptr<global, T>, dim_0..dim_{r-1}, stride_0..stride_{r-1}`.
// Returns the `(ptr, rank)` pair on success.
static std::optional<std::pair<Value, unsigned>>
validateKernelArgCastShape(UnrealizedConversionCastOp cast) {
  if (cast.getOutputs().size() != 1 || cast.getInputs().size() < 1 ||
      (cast.getInputs().size() - 1) % 2 != 0)
    return std::nullopt;
  Value ptr = cast.getInputs().front();
  auto ptrType = dyn_cast<PtrType>(ptr.getType());
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Global)
    return std::nullopt;
  unsigned rank = static_cast<unsigned>((cast.getInputs().size() - 1) / 2);
  return std::make_pair(ptr, rank);
}

// Extract `rank` `index`-typed values from `cast.getInputs()`
// starting at `start`. Returns `nullopt` if any value isn't `index`.
static std::optional<SmallVector<Value>>
extractKernelArgIndexInputs(UnrealizedConversionCastOp cast, unsigned start,
                            unsigned rank) {
  SmallVector<Value> out;
  out.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value v = cast.getInputs()[start + axis];
    if (!v.getType().isIndex())
      return std::nullopt;
    out.push_back(v);
  }
  return out;
}

static std::optional<KernelArgSource> resolveKernelArg(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast)
    return std::nullopt;

  if (std::optional<KernelArgSource> via =
          resolveKernelArgViaBareCarrier(cast, source))
    return via;

  std::optional<std::pair<Value, unsigned>> shape =
      validateKernelArgCastShape(cast);
  if (!shape)
    return std::nullopt;
  auto [ptr, rank] = *shape;
  std::optional<SmallVector<Value>> dims =
      extractKernelArgIndexInputs(cast, /*start=*/1, rank);
  if (!dims)
    return std::nullopt;
  std::optional<SmallVector<Value>> strides =
      extractKernelArgIndexInputs(cast, /*start=*/1 + rank, rank);
  if (!strides)
    return std::nullopt;

  KernelArgSource info;
  info.ptr = ptr;
  info.dims = std::move(*dims);
  info.strides = std::move(*strides);
  return info;
}

// True when `source` is the rank-1 carrier output of the post-flatten
// retype UCC (`buffer<T, [...]> -> buffer<T, ["?"]>, idx..., idx...`).
// Together with `resolveKernelArg` (which transparently recurses
// through the retype to the kernel-arg bundle), this lets a consumer
// distinguish "kernel-arg accessed via its native rank-N indexing" from
// "kernel-arg accessed via a single composed 1D offset over the post-
// flatten carrier".
static bool isPostFlattenKernelArgSource(Value source) {
  auto cast = source.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1 ||
      source != cast.getOutputs()[0])
    return false;
  auto bufOut = dyn_cast<BufferType>(source.getType());
  return bufOut && bufOut.getShape().getDims().size() == 1;
}

// IR-mutating helper: synthesize a rank-1 kernel-arg view (same ptr,
// placeholder dim, unit stride). Used by consumers that need a uniform
// rank-1 surface when the access uses a single composed offset against
// the post-flatten carrier. The constants land at the builder's
// insertion point; downstream canonicalize/cse folds the placeholder
// dim's `arith.constant 0` away because nothing uses it.
static KernelArgSource flatKernelArgView(OpBuilder &builder, Location loc,
                                         const KernelArgSource &inner) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value one = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  KernelArgSource flat;
  flat.ptr = inner.ptr;
  flat.dims = {zero};
  flat.strides = {one};
  return flat;
}

// Resolve `source` to a kernel-arg view shaped to match how the
// access op is using the buffer: pre-flatten rank-N access against a
// rank-N kernel-arg → rank-N view; post-flatten rank-1 access against
// the same kernel-arg (reached through the flatten retype UCC) →
// rank-1 view synthesized via `flatKernelArgView`. The pre-flatten
// shorthand (single UCC carrying a rank-1 carrier) falls through the
// rank-N branch with rank already == 1, so consumers don't have to
// special-case it.
static std::optional<KernelArgSource>
resolveAccessKernelArg(OpBuilder &builder, Location loc, Value source) {
  auto info = resolveKernelArg(source);
  if (!info)
    return std::nullopt;
  if (isPostFlattenKernelArgSource(source) && info->rank() > 1)
    return flatKernelArgView(builder, loc, *info);
  return info;
}

// Compose the linear element offset for a multi-axis access against a
// kernel-arg buffer: `sum_axis(indices[axis] * strides[axis])`. The result
// is in element units (matching torch/numpy `tensor.stride(i)` reporting),
// not bytes — `hc.ptr_offset` does its own element-size scaling on the
// way to LLVM.
static Value linearizeKernelArgOffset(OpBuilder &builder, Location loc,
                                      const KernelArgSource &source,
                                      ValueRange indices) {
  assert(indices.size() == source.rank() &&
         "kernel-arg offset rank must match source rank");
  Value offset;
  for (auto [index, stride] : llvm::zip_equal(indices, source.strides)) {
    Value term = arith::MulIOp::create(builder, loc, index, stride).getResult();
    offset = offset
                 ? arith::AddIOp::create(builder, loc, offset, term).getResult()
                 : term;
  }
  if (!offset)
    offset = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  return offset;
}

static Value indexCast(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), value)
      .getResult(0);
}

// Bind each free shape symbol from `type` to its matching dim value pulled
// out of the kernel-arg UCC fragment. Replaces the prior `memref.dim`
// chain — the dims now ride as explicit UCC inputs (one per axis).
static void bindShapeSymbols(BufferType type, const KernelArgSource &source,
                             BoundValues &boundValues) {
  for (auto [axis, attr] : llvm::enumerate(type.getShape().getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<StringRef> symbol = exactSymbolName(expr);
    if (!symbol)
      continue;
    if (axis >= source.rank())
      continue;
    boundValues.bind(*symbol, source.dims[axis]);
  }
}

// Map a sym name produced by the frontend's default strided layout
// (`$STRIDE_<axis>_<argname>`, per `buildDefaultStridedBufferLayout`)
// to the corresponding kernel-arg `index`-typed stride value. Returns
// null when `symName` doesn't match the convention or the axis is
// out of range. Pairs with the shape-dim walk below to cover every
// implicit sym a kernel-arg bundle carries.
static Value resolveBundleStrideSym(const KernelArgSource &source,
                                    StringRef symName) {
  if (!symName.consume_front("$STRIDE_"))
    return Value{};
  unsigned axis = 0;
  if (symName.consumeInteger(10, axis))
    return Value{};
  if (!symName.starts_with("_"))
    return Value{};
  if (axis >= source.strides.size())
    return Value{};
  return source.strides[axis];
}

// If `value` is an `!hc.idx<sym>` whose defining op is a post-flatten
// retype UCC chaining back to a kernel-arg bundle, return the
// bundle's corresponding `index`-typed input value (shape dim or
// stride per the frontend's default strided layout convention).
// Returns null when the chain doesn't reach a kernel-arg bundle, or
// when the sym doesn't map to any kernel-ABI slot (layout-param
// syms, opaque non-bundle indices, etc.).
//
// Binding to the bundle root short-circuits the post-flatten retype
// UCC + `idx<sym>→index` cast that `reconcileUnrealizedCasts` can't
// reduce across (the HC-typed intermediate hides the round trip from
// MLIR's standard UCC-folding view, and the leftover UCC fails LLVM
// translation downstream). The kernel-arg bundle's `index` inputs
// fold through the `index → i64` block-arg materialization that
// `convert-gpu-to-rocdl` plants, taking the offset arith all the way
// back to the i64 kernel args.
// Walk the bundle's per-axis shape syms looking for a dim that
// matches `symbol`. Returns the bundle's `index`-typed dim value
// when found.
static std::optional<Value> lookupBundleDim(BufferType bundleType,
                                            const KernelArgSource &info,
                                            StringRef symbol) {
  for (auto [axis, attr] : llvm::enumerate(bundleType.getShape().getDims())) {
    auto expr = dyn_cast<ExprAttr>(attr);
    std::optional<StringRef> dimSym = exactSymbolName(expr);
    if (dimSym && *dimSym == symbol && axis < info.dims.size())
      return info.dims[axis];
  }
  return std::nullopt;
}

static Value resolveToBundleIndex(Value value) {
  auto idxType = dyn_cast<IdxType>(value.getType());
  if (!idxType)
    return Value{};
  std::optional<StringRef> symbol = exactSymbolName(idxType.getExpr());
  if (!symbol)
    return Value{};
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast.getOutputs().size() <= 1)
    return Value{};
  auto bundleType = dyn_cast<BufferType>(cast.getInputs()[0].getType());
  if (!bundleType)
    return Value{};
  std::optional<KernelArgSource> info = resolveKernelArg(cast.getInputs()[0]);
  if (!info)
    return Value{};
  if (std::optional<Value> dim = lookupBundleDim(bundleType, *info, *symbol))
    return *dim;
  return resolveBundleStrideSym(*info, *symbol);
}

// `indexCast` with a short-circuit through the kernel-arg bundle: if
// `value` chains back to one, use the bundle's `index`-typed input
// directly; otherwise fall back to emitting an `idx<sym>→index` UCC.
static Value indexCastViaBundle(OpBuilder &builder, Location loc, Value value) {
  if (Value direct = resolveToBundleIndex(value))
    return direct;
  return indexCast(builder, loc, value);
}

static void bindLaunchDim3(StringRef prefix, gpu::KernelDim3 values,
                           BoundValues &boundValues) {
  Value dims[] = {values.x, values.y, values.z};
  for (auto [axis, value] : llvm::enumerate(dims)) {
    SmallString<16> name(prefix);
    name += Twine(axis).str();
    boundValues.bind(name, value);
  }
}

// `$WO[k] = $WG[k] * $WGS[k]`: the upper-left corner of the
// workgroup's tile in the work grid, on axis `k`. The launch
// already carries both factors as `index`-typed operands, so the
// binding is one `arith.muli` per axis, planted at the rewriter's
// current insertion point. CSE collapses the per-apply duplicates
// downstream — keeping the materialisation here (rather than
// substituting `$WO` -> `$WG*$WGS` symbolically before lowering)
// matches the bind-then-lookup shape of every other launch-geometry
// sym and leaves the symbol equivalence intact for any future
// consumer that cares about the name.
static void bindWorkOffsets(gpu::LaunchOp launch, OpBuilder &builder,
                            Location loc, BoundValues &boundValues) {
  gpu::KernelDim3 blockIds = launch.getBlockIds();
  gpu::KernelDim3 blockSizes = launch.getBlockSizeOperandValues();
  Value bi[] = {blockIds.x, blockIds.y, blockIds.z};
  Value bs[] = {blockSizes.x, blockSizes.y, blockSizes.z};
  for (auto axis : llvm::seq<size_t>(0, std::size(bi))) {
    SmallString<8> name("$WO");
    name += Twine(axis).str();
    Value product = arith::MulIOp::create(builder, loc, bi[axis], bs[axis]);
    boundValues.bind(name, product);
  }
}

// Pre-flatten kernel-arg UCC: single multi-input bundle → single
// buffer output. The buffer carries shape syms (M, N, ...) and the
// inputs give us per-axis dim values.
static void bindPreFlattenKernelArgCast(UnrealizedConversionCastOp cast,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        BoundValues &boundValues) {
  Type outputType = cast.getOutputs().front().getType();
  if (auto buffer = dyn_cast<BufferType>(outputType)) {
    if (auto info = resolveKernelArg(cast.getResult(0)))
      bindShapeSymbols(buffer, *info, boundValues);
    return;
  }
  if (cast.getInputs().size() != 1)
    return;
  Value input = cast.getInputs().front();
  if (std::optional<StringRef> symbol = exactSymbolName(outputType))
    boundValues.bind(*symbol, indexCast(rewriter, loc, input));
}

// Post-flatten retype UCC: rank-N buffer input → (rank-1 buffer,
// idx<sym>, idx<sym>, ...) outputs. Each idx-typed output exposes
// an implicit-symbol value (kernel-arg shape dim, stride, layout
// param, ...) that the access ops reference symbolically in their
// composed offsets. Bind each one so the ambient lowering can
// resolve the apply.
//
// `indexCastViaBundle` short-circuits to the underlying kernel-arg
// bundle's `index`-typed input when one is reachable; otherwise it
// emits an `idx<sym>→index` UCC on the retype output as before. See
// `resolveToBundleIndex` for the rationale.
static void bindPostFlattenRetypeCast(UnrealizedConversionCastOp cast,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc, BoundValues &boundValues) {
  if (cast.getInputs().size() != 1)
    return;
  for (Value output : cast.getOutputs().drop_front()) {
    std::optional<StringRef> symbol = exactSymbolName(output.getType());
    if (symbol)
      boundValues.bind(*symbol, indexCastViaBundle(rewriter, loc, output));
  }
}

static void bindKernelArgCastSymbols(UnrealizedConversionCastOp cast,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, BoundValues &boundValues) {
  if (cast.getOutputs().size() == 1) {
    bindPreFlattenKernelArgCast(cast, rewriter, loc, boundValues);
    return;
  }
  bindPostFlattenRetypeCast(cast, rewriter, loc, boundValues);
}

// Structured loop/region block arguments that carry a bare-sym `!hc.idx`
// type (e.g. an `hc.for_range` induction variable typed
// `!hc.idx<"$join0">`) are their own binding for that symbol name. The
// pre-flatten access-offset apply leaves these as free symbols because
// the inner expression references the bare sym directly rather than
// taking the value as an explicit operand; we pick them up ambiently
// here. Only ancestor blocks of `anchor` qualify — a sibling for_range's
// induction var would shadow incorrectly and would also fail SSA
// dominance if a UCC against it ended up outside its defining block.
//
// Post-`hc-flatten-with-layouts`, `hc.generic` carries its ambient sym
// SSA edges in operand form (`ambient_idxs` / `ambient_idx_syms`) and
// `hc-lower-generic` plants per-lane applies that bind every ambient
// sym as an explicit operand. Those applies never reach this walker
// because their operand list is complete; the only consumers we see
// are pre-flatten applies whose free-sym set still rides on the
// ancestor `!hc.idx<sym>` types we collect below.
static void bindAncestorBlockArgSymbols(Operation *anchor, gpu::LaunchOp launch,
                                        ConversionPatternRewriter &rewriter,
                                        BoundValues &boundValues) {
  for (Block *block = anchor->getBlock(); block;) {
    for (BlockArgument arg : block->getArguments()) {
      std::optional<StringRef> symbol = exactSymbolName(arg.getType());
      if (!symbol)
        continue;
      boundValues.bind(*symbol, indexCast(rewriter, anchor->getLoc(), arg));
    }
    Operation *parent = block->getParentOp();
    if (!parent || parent == launch.getOperation())
      break;
    block = parent->getBlock();
  }
}

static BoundValues collectBoundValues(Operation *anchor,
                                      ConversionPatternRewriter &rewriter) {
  BoundValues boundValues;
  auto launch = anchor->getParentOfType<gpu::LaunchOp>();
  if (!launch)
    return boundValues;

  bindLaunchDim3("$WG", launch.getBlockIds(), boundValues);
  bindLaunchDim3("$WI", launch.getThreadIds(), boundValues);
  bindLaunchDim3("$WGS", launch.getBlockSizeOperandValues(), boundValues);
  bindWorkOffsets(launch, rewriter, anchor->getLoc(), boundValues);

  launch.walk([&](UnrealizedConversionCastOp cast) {
    bindKernelArgCastSymbols(cast, rewriter, anchor->getLoc(), boundValues);
  });

  bindAncestorBlockArgSymbols(anchor, launch, rewriter, boundValues);

  return boundValues;
}

class ExprLowerer {
public:
  ExprLowerer(OpBuilder &builder, Location loc, const BoundValues &boundValues)
      : builder(builder), loc(loc), boundValues(boundValues) {}

  FailureOr<Value> lower(ExprAttr expr) {
    if (!expr)
      return failure();
    return lowerNode(rawNode(expr));
  }

  FailureOr<Value> lower(PredAttr pred) {
    if (!pred)
      return failure();
    return lowerPredNode(rawNode(pred));
  }

  Value constant(int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value).getResult();
  }

private:
  FailureOr<Value> lowerNode(ixs_node *node) {
    switch (ixs_node_tag(node)) {
    case IXS_INT:
      return constant(ixs_node_int_val(node));
    case IXS_RAT:
      return lowerRational(node);
    case IXS_SYM:
      return lowerSymbol(node);
    case IXS_ADD:
      return lowerAdd(node);
    case IXS_MUL:
      return lowerMul(node);
    case IXS_CEIL:
      return lowerCeil(node);
    case IXS_FLOOR:
      return lowerNode(ixs_node_unary_arg(node));
    case IXS_MOD:
      return lowerBinary<arith::RemUIOp>(node);
    default:
      return failure();
    }
  }

  FailureOr<Value> lowerPredNode(ixs_node *node) {
    switch (ixs_node_tag(node)) {
    case IXS_TRUE:
      return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true))
          .getResult();
    case IXS_FALSE:
      return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(false))
          .getResult();
    case IXS_CMP:
      return lowerCmp(node);
    case IXS_AND:
      return lowerLogic<arith::AndIOp>(node);
    case IXS_OR:
      return lowerLogic<arith::OrIOp>(node);
    case IXS_NOT: {
      FailureOr<Value> value = lowerPredNode(ixs_node_unary_arg(node));
      if (failed(value))
        return failure();
      Value trueValue =
          arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
      return arith::XOrIOp::create(builder, loc, *value, trueValue).getResult();
    }
    default:
      return failure();
    }
  }

  FailureOr<Value> lowerRational(ixs_node *node) {
    int64_t numerator = ixs_node_rat_num(node);
    int64_t denominator = ixs_node_rat_den(node);
    if (denominator == 1)
      return constant(numerator);
    return failure();
  }

  FailureOr<Value> lowerSymbol(ixs_node *node) {
    StringRef name(ixs_node_sym_name(node));
    if (Value value = boundValues.lookup(name))
      return value;
    // Record the first miss so the caller can emit a diagnostic
    // naming the unresolved symbol. We don't emit here because
    // ExprLowerer is a per-node walker without access to the op
    // surface that should own the diagnostic. The caller checks
    // `lastUnresolvedSymbol()` on failure.
    if (unresolvedSymbol.empty())
      unresolvedSymbol = name.str();
    return failure();
  }

  FailureOr<Value> lowerAdd(ixs_node *node) {
    FailureOr<Value> result = lowerNode(ixs_node_add_coeff(node));
    if (failed(result))
      return failure();

    for (uint32_t index = 0, end = ixs_node_add_nterms(node); index != end;
         ++index) {
      FailureOr<Value> term = lowerNode(ixs_node_add_term(node, index));
      if (failed(term))
        return failure();
      FailureOr<Value> scaled =
          scaleTerm(ixs_node_add_term_coeff(node, index), *term);
      if (failed(scaled))
        return failure();
      *result = arith::AddIOp::create(builder, loc, *result, *scaled);
    }
    return *result;
  }

  FailureOr<Value> scaleTerm(ixs_node *coeff, Value term) {
    if (isOneNode(coeff))
      return term;
    if (ixs_node_tag(coeff) == IXS_INT) {
      Value coeffValue = constant(ixs_node_int_val(coeff));
      return arith::MulIOp::create(builder, loc, coeffValue, term).getResult();
    }
    if (ixs_node_tag(coeff) != IXS_RAT)
      return failure();

    int64_t numerator = ixs_node_rat_num(coeff);
    int64_t denominator = ixs_node_rat_den(coeff);
    if (denominator <= 0)
      return failure();
    Value scaled = term;
    if (numerator != 1)
      scaled = arith::MulIOp::create(builder, loc, constant(numerator), term);
    if (denominator != 1)
      scaled =
          arith::DivUIOp::create(builder, loc, scaled, constant(denominator));
    return scaled;
  }

  FailureOr<Value> lowerMul(ixs_node *node) {
    FailureOr<std::pair<Value, Value>> fraction = lowerAsFraction(node);
    if (failed(fraction))
      return failure();
    if (auto denominator =
            fraction->second.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<IntegerAttr>(denominator.getValue()))
        if (attr.getInt() == 1)
          return fraction->first;
    }
    return arith::DivUIOp::create(builder, loc, fraction->first,
                                  fraction->second)
        .getResult();
  }

  FailureOr<Value> lowerCeil(ixs_node *node) {
    FailureOr<std::pair<Value, Value>> fraction =
        lowerAsFraction(ixs_node_unary_arg(node));
    if (failed(fraction))
      return failure();
    return arith::CeilDivUIOp::create(builder, loc, fraction->first,
                                      fraction->second)
        .getResult();
  }

  FailureOr<std::pair<Value, Value>> lowerAsFraction(ixs_node *node) {
    if (ixs_node_tag(node) != IXS_MUL) {
      FailureOr<Value> value = lowerNode(node);
      if (failed(value))
        return failure();
      return std::pair<Value, Value>{*value, constant(1)};
    }

    ixs_node *coeff = ixs_node_mul_coeff(node);
    int64_t numerator = 1;
    int64_t denominator = 1;
    if (ixs_node_tag(coeff) == IXS_INT) {
      numerator = ixs_node_int_val(coeff);
    } else if (ixs_node_tag(coeff) == IXS_RAT) {
      numerator = ixs_node_rat_num(coeff);
      denominator = ixs_node_rat_den(coeff);
    } else {
      return failure();
    }
    if (denominator <= 0)
      return failure();

    Value numeratorValue = constant(numerator);
    for (uint32_t index = 0, end = ixs_node_mul_nfactors(node); index != end;
         ++index) {
      int32_t exponent = ixs_node_mul_factor_exp(node, index);
      if (exponent < 0)
        return failure();
      FailureOr<Value> factor =
          lowerNode(ixs_node_mul_factor_base(node, index));
      if (failed(factor))
        return failure();
      for (int32_t power = 0; power < exponent; ++power)
        numeratorValue =
            arith::MulIOp::create(builder, loc, numeratorValue, *factor);
    }
    return std::pair<Value, Value>{numeratorValue, constant(denominator)};
  }

  template <typename OpT> FailureOr<Value> lowerBinary(ixs_node *node) {
    FailureOr<Value> lhs = lowerNode(ixs_node_binary_lhs(node));
    FailureOr<Value> rhs = lowerNode(ixs_node_binary_rhs(node));
    if (failed(lhs) || failed(rhs))
      return failure();
    return OpT::create(builder, loc, *lhs, *rhs).getResult();
  }

  FailureOr<Value> lowerCmp(ixs_node *node) {
    FailureOr<Value> lhs = lowerNode(ixs_node_binary_lhs(node));
    FailureOr<Value> rhs = lowerNode(ixs_node_binary_rhs(node));
    if (failed(lhs) || failed(rhs))
      return failure();
    arith::CmpIPredicate pred = arith::CmpIPredicate::eq;
    switch (ixs_node_cmp_op(node)) {
    case IXS_CMP_LT:
      pred = arith::CmpIPredicate::slt;
      break;
    case IXS_CMP_LE:
      pred = arith::CmpIPredicate::sle;
      break;
    case IXS_CMP_GT:
      pred = arith::CmpIPredicate::sgt;
      break;
    case IXS_CMP_GE:
      pred = arith::CmpIPredicate::sge;
      break;
    case IXS_CMP_EQ:
      pred = arith::CmpIPredicate::eq;
      break;
    case IXS_CMP_NE:
      pred = arith::CmpIPredicate::ne;
      break;
    }
    return arith::CmpIOp::create(builder, loc, pred, *lhs, *rhs).getResult();
  }

  template <typename OpT> FailureOr<Value> lowerLogic(ixs_node *node) {
    uint32_t count = ixs_node_logic_nargs(node);
    if (count == 0)
      return failure();
    FailureOr<Value> result = lowerPredNode(ixs_node_logic_arg(node, 0));
    if (failed(result))
      return failure();
    for (uint32_t index = 1; index != count; ++index) {
      FailureOr<Value> arg = lowerPredNode(ixs_node_logic_arg(node, index));
      if (failed(arg))
        return failure();
      *result = OpT::create(builder, loc, *result, *arg).getResult();
    }
    return *result;
  }

  OpBuilder &builder;
  Location loc;
  const BoundValues &boundValues;
  std::string unresolvedSymbol;

public:
  // First symbol name that lowerSymbol couldn't bind during the most
  // recent `lower(...)` call. Empty if every reachable symbol was
  // bound. Callers use this to plant a diagnostic that names the
  // missing sym instead of the generic "failed to lower" message,
  // matching the contract in `doc/layouts.md` "Free symbols in
  // layout offsets" (validation is delayed to lowering, and the
  // error has to identify the offending name).
  StringRef lastUnresolvedSymbol() const { return unresolvedSymbol; }
};

static Type convertIntrinsicBoundaryType(Type type,
                                         const TypeConverter &converter) {
  if (isa<BareTensorType, BareVectorType>(type))
    return type;
  auto tuple = dyn_cast<TupleType>(type);
  if (!tuple)
    return converter.convertType(type);

  SmallVector<Type> elements;
  elements.reserve(tuple.size());
  for (Type element : tuple.getTypes()) {
    Type converted = convertIntrinsicBoundaryType(element, converter);
    if (!converted)
      return {};
    elements.push_back(converted);
  }
  return TupleType::get(type.getContext(), elements);
}

static LogicalResult convertIntrinsicFunctionSignature(
    FunctionType fnType, const TypeConverter &converter,
    SmallVectorImpl<Type> &inputs, SmallVectorImpl<Type> &results) {
  for (Type input : fnType.getInputs()) {
    Type converted = convertIntrinsicBoundaryType(input, converter);
    if (!converted)
      return failure();
    inputs.push_back(converted);
  }
  for (Type result : fnType.getResults()) {
    Type converted = convertIntrinsicBoundaryType(result, converter);
    if (!converted)
      return failure();
    results.push_back(converted);
  }
  return success();
}

static LogicalResult convertIntrinsicBodySignature(
    FunctionType fnType, TypeConverter::SignatureConversion &bodyConversion,
    const TypeConverter &converter) {
  for (auto [index, input] : llvm::enumerate(fnType.getInputs())) {
    Type converted = convertIntrinsicBoundaryType(input, converter);
    if (!converted)
      return failure();
    bodyConversion.addInputs(index, converted);
  }
  return success();
}

struct ConvertIntrinsicSignatureOp : public OpConversionPattern<HCIntrinsicOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIntrinsicOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<FunctionType> fnType = op.getFunctionType();
    if (!fnType)
      return failure();

    SmallVector<Type> inputs;
    SmallVector<Type> results;
    if (failed(convertIntrinsicFunctionSignature(*fnType, *typeConverter,
                                                 inputs, results)))
      return failure();

    auto converted = FunctionType::get(rewriter.getContext(), inputs, results);
    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(converted);
      if (!op.getBody().empty()) {
        TypeConverter::SignatureConversion bodyConversion(
            fnType->getNumInputs());
        (void)convertIntrinsicBodySignature(*fnType, bodyConversion,
                                            *typeConverter);
        rewriter.applySignatureConversion(&op.getBody().front(), bodyConversion,
                                          typeConverter);
      }
    });
    return success();
  }
};

// Build a `BoundValues` for one `hc.idx_apply` / `hc.pred_apply` op.
// Explicit operand bindings are authoritative (direct map insert
// rather than `bind()`, which uses `try_emplace`); the ambient walk
// fills in everything else for free symbols left unlisted (e.g.
// launch geometry like `$WG0`).
static BoundValues collectApplyBindings(Operation *op,
                                        ConversionPatternRewriter &rewriter,
                                        ArrayAttr symbols,
                                        ValueRange convertedOperands) {
  BoundValues boundValues = collectBoundValues(op, rewriter);
  for (auto [attr, operand] : llvm::zip(symbols, convertedOperands)) {
    StringRef name = cast<StringAttr>(attr).getValue();
    if (name.empty())
      continue;
    boundValues.symbols[name] =
        indexCastViaBundle(rewriter, op->getLoc(), operand);
  }
  return boundValues;
}

struct ConvertIdxApplyOp : public OpConversionPattern<HCIdxApplyOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIdxApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto idx = dyn_cast<IdxType>(op.getResult().getType());
    if (!idx || !idx.getExpr())
      return op.emitOpError("expected `!hc.idx<expr>` result type");

    BoundValues boundValues = collectApplyBindings(
        op, rewriter, op.getSymbolsAttr(), adaptor.getOperands());
    ExprLowerer lowerer(rewriter, op.getLoc(), boundValues);
    FailureOr<Value> lowered = lowerer.lower(idx.getExpr());
    if (failed(lowered)) {
      StringRef sym = lowerer.lastUnresolvedSymbol();
      if (!sym.empty())
        return op.emitOpError("cannot lower idx_apply: free symbol '")
               << sym << "' has no binding in the surrounding scope "
               << "(launch geometry, kernel-arg aux idx, or ancestor block "
                  "argument); supply it via the access op's aux operands or "
                  "the producing op's binding slot";
      return op.emitOpError("failed to lower idx_apply expression");
    }
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

struct ConvertPredApplyOp : public OpConversionPattern<HCPredApplyOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPredApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred = dyn_cast<PredType>(op.getResult().getType());
    if (!pred || !pred.getPred())
      return op.emitOpError("expected `!hc.pred<pred>` result type");

    BoundValues boundValues = collectApplyBindings(
        op, rewriter, op.getSymbolsAttr(), adaptor.getOperands());
    ExprLowerer lowerer(rewriter, op.getLoc(), boundValues);
    FailureOr<Value> lowered = lowerer.lower(pred.getPred());
    if (failed(lowered)) {
      StringRef sym = lowerer.lastUnresolvedSymbol();
      if (!sym.empty())
        return op.emitOpError("cannot lower pred_apply: free symbol '")
               << sym << "' has no binding in the surrounding scope";
      return op.emitOpError("failed to lower pred_apply predicate");
    }
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

static FailureOr<TypedAttr> splatAttr(OpBuilder &builder, Type type,
                                      int64_t value) {
  Type elementType = type;
  if (auto shaped = dyn_cast<ShapedType>(type))
    elementType = shaped.getElementType();

  Attribute scalar;
  if (elementType.isInteger(1))
    scalar = builder.getBoolAttr(value != 0);
  else if (elementType.isIndex())
    scalar = builder.getIndexAttr(value);
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    scalar = builder.getIntegerAttr(intType, value);
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    scalar = builder.getFloatAttr(floatType, value);
  else
    return failure();

  if (auto shaped = dyn_cast<ShapedType>(type))
    return cast<TypedAttr>(DenseElementsAttr::get(shaped, scalar));
  return cast<TypedAttr>(scalar);
}

struct ConvertConstOp : public OpConversionPattern<HCConstOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCConstOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();

    Attribute value = op.getValue();
    if (converted.isIndex()) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return op.emitOpError("expected integer literal for index constant");
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, integer.getInt());
      return success();
    }

    if (auto typed = dyn_cast<TypedAttr>(value)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, converted, typed);
      return success();
    }
    return failure();
  }
};

template <typename OpT, typename ArithOpT>
struct ConvertIntBinaryOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted || !converted.isIntOrIndex())
      return failure();
    rewriter.template replaceOpWithNewOp<ArithOpT>(
        op, converted, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// Float counterpart to `ConvertIntBinaryOp`. The per-element scalar body
// `hc-elementwise-to-generic` plants inside `hc.generic` carries the source
// op kind (`hc.add` / `hc.sub` / ...) regardless of element type — the int
// pattern above bails on float-typed bodies and this one picks them up.
// The two together cover the numeric surface; pred/index/cast handlers
// elsewhere cover the rest.
template <typename OpT, typename ArithOpT>
struct ConvertFloatBinaryOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted || !isa<FloatType>(converted))
      return failure();
    rewriter.template replaceOpWithNewOp<ArithOpT>(
        op, converted, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SliceAxis {
  Value offset;
  Value stride;
  bool isSlice = false;
};

static Value zeroIndex(OpBuilder &builder, Location loc) {
  return arith::ConstantIndexOp::create(builder, loc, 0).getResult();
}

static Value oneIndex(OpBuilder &builder, Location loc) {
  return arith::ConstantIndexOp::create(builder, loc, 1).getResult();
}

static LogicalResult collectSliceAxis(Operation *op, Value index,
                                      OpBuilder &builder, SliceAxis &axis,
                                      bool requireUnitStride = true) {
  auto slice = index.getDefiningOp<HCSliceExprOp>();
  if (!slice)
    return failure();
  if (Value step = slice.getStep()) {
    APInt stepValue;
    if (requireUnitStride && (!matchPattern(step, m_ConstantInt(&stepValue)) ||
                              stepValue.getSExtValue() != 1))
      return op->emitOpError("only unit-stride slices lower to vector ops");
    axis.stride = step;
  } else {
    axis.stride = oneIndex(builder, op->getLoc());
  }
  axis.offset =
      slice.getLower() ? slice.getLower() : zeroIndex(builder, op->getLoc());
  axis.isSlice = true;
  return success();
}

static FailureOr<SmallVector<SliceAxis>>
collectAxes(Operation *op, ValueRange indices, OpBuilder &builder,
            bool requireUnitStride = true) {
  SmallVector<SliceAxis> axes;
  axes.reserve(indices.size());
  for (Value index : indices) {
    SliceAxis axis;
    if (isa<SliceType>(index.getType())) {
      if (failed(collectSliceAxis(op, index, builder, axis, requireUnitStride)))
        return failure();
    } else {
      if (!index.getType().isIndex())
        return op->emitOpError("expected index or slice subscript after "
                               "launch-body scalar lowering");
      axis.offset = index;
      axis.stride = oneIndex(builder, op->getLoc());
    }
    axes.push_back(axis);
  }
  return axes;
}

// Post-flatten access-op shape: one scalar `index` subscript carrying the
// already-linearised element offset, against a 1-rank kernel-arg ABI whose
// only stride is constant 1, and a 1-rank iter shape from the result vector
// (load/vload), source vector (store), or mask vector (load_mask). The
// pre-flatten per-axis structure has been folded into the composed offset
// by `hc-flatten-with-layouts`; the remaining lane walk over the flat tile
// is a unit-stride bump over `[composed_offset, composed_offset + N)`.
//
// Synthesizing a single full-slice axis (`offset = composed`, `stride = 1`,
// `isSlice = true`) lets the existing per-lane machinery in
// `kernelArgLaneIndices` / `storeIndicesForCoordinate` /
// `linearizeKernelArgOffset` fall through unchanged: the kernel-arg's
// unit stride collapses through the lin formula and each lane's offset
// reduces to `composed + lane`. Non-contiguous post-flatten tiles
// (strided slice survivors) intentionally fail this detect and route
// through the slice-aware path — they need the rank-N kernel-arg ABI
// to recompute per-axis offsets and are tracked separately.
static std::optional<SmallVector<SliceAxis>>
synthesizePostFlattenAxes(OpBuilder &builder, Location loc,
                          const KernelArgSource &kernelArg, ValueRange indices,
                          ArrayRef<int64_t> iterShape) {
  if (kernelArg.rank() != 1)
    return std::nullopt;
  if (indices.size() != 1)
    return std::nullopt;
  if (iterShape.size() != 1)
    return std::nullopt;
  if (!indices[0].getType().isIndex())
    return std::nullopt;
  APInt stride;
  if (!matchPattern(kernelArg.strides[0], m_ConstantInt(&stride)) ||
      stride.getSExtValue() != 1)
    return std::nullopt;
  SliceAxis axis;
  axis.offset = indices[0];
  axis.stride = oneIndex(builder, loc);
  axis.isSlice = true;
  return SmallVector<SliceAxis>{axis};
}

// Allocate a flat `!hc.ptr<workgroup, T>` sized for `count` elements. The
// downstream `hc-lower-to-llvm` rewrites the alloc into a sibling
// addrspace(3) global; rank-erasure happens at this boundary because the
// AMDGPU LDS surface is one-dimensional anyway.
static Value allocateWorkgroupPtr(OpBuilder &builder, Location loc,
                                  PtrType ptrType, int64_t count) {
  Value countValue =
      arith::ConstantIndexOp::create(builder, loc, count).getResult();
  return HCAllocOp::create(builder, loc, ptrType, countValue).getResult();
}

// Walk linear `lin` into per-axis coords for `shape` (last axis fastest).
// Returns the constant-index `Value` per axis.
static SmallVector<Value> unlinearizeCoords(OpBuilder &builder, Location loc,
                                            int64_t lin,
                                            ArrayRef<int64_t> shape) {
  SmallVector<Value> coords(shape.size());
  int64_t remaining = lin;
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
       --axis) {
    coords[axis] =
        arith::ConstantIndexOp::create(builder, loc, remaining % shape[axis])
            .getResult();
    remaining /= shape[axis];
  }
  return coords;
}

// Per-element store from a multi-dim `vector<...xT>` value into a flat
// `!hc.ptr<workgroup, T>` buffer. Emits one `hc.ptr_offset` + `hc.ptr_store`
// per lane in lex order (rightmost axis varies fastest). We
// unconditionally use the per-element form
// (rather than a single vector-typed `hc.ptr_store`) so the matching reads
// can also be per-element scalars and the i1 byte-vs-bit discrepancy LLVM
// has between scalar and vector i1 stores never bites.
static LogicalResult writeVectorToWorkgroupPtr(OpBuilder &builder, Location loc,
                                               Value vector, Value ptr,
                                               ArrayRef<int64_t> shape) {
  auto vectorType = dyn_cast<mlir::VectorType>(vector.getType());
  if (!vectorType || vectorType.getShape() != shape)
    return failure();
  auto ptrType = dyn_cast<PtrType>(ptr.getType());
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();

  int64_t total = 1;
  for (int64_t d : shape)
    total *= d;
  for (int64_t lin = 0; lin != total; ++lin) {
    SmallVector<Value> coords = unlinearizeCoords(builder, loc, lin, shape);
    SmallVector<int64_t> coordInts(shape.size());
    int64_t remaining = lin;
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
         --axis) {
      coordInts[axis] = remaining % shape[axis];
      remaining /= shape[axis];
    }
    Value element =
        vector::ExtractOp::create(builder, loc, vector, coordInts).getResult();
    Value linVal =
        arith::ConstantIndexOp::create(builder, loc, lin).getResult();
    Value addr =
        HCPtrOffsetOp::create(builder, loc, ptrType, ptr, linVal).getResult();
    HCPtrStoreOp::create(builder, loc, element, addr);
  }
  return success();
}

// Walk back through an `unrealized_conversion_cast` to recover a workgroup-AS
// `!hc.ptr` value. Counterpart to `resolveKernelArg` for the launch-body's
// LDS path; kernel-arg buffers come in as ptr+dims+strides UCC bundles
// for now. Returns null if `value` doesn't ultimately derive from a
// workgroup ptr.
static Value sourcePtr(Value value) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (cast && cast.getInputs().size() == 1 && cast.getOutputs().size() == 1) {
    Value input = cast.getInputs().front();
    if (auto ptr = dyn_cast<PtrType>(input.getType()))
      if (ptr.getAddrSpace() == AddrSpace::Workgroup)
        return input;
  }
  if (auto ptr = dyn_cast<PtrType>(value.getType()))
    if (ptr.getAddrSpace() == AddrSpace::Workgroup)
      return value;
  return {};
}

// Captures the underlying `!hc.ptr<workgroup, T>` source plus an indexing
// pattern that maps result-vector lanes back to source-tile flat offsets.
// Materialized eagerly from a (possibly view-laden) bare-tensor SSA value;
// see `resolvePtrViewSource`. The trivial pattern (no `hc.buffer_view` in
// the chain) carries one full-slice `SliceAxis` per source dim — gives the
// per-element loop a single shape to iterate without a separate "no view"
// path.
struct PtrViewSource {
  Value sourcePtr;
  PtrType ptrType;
  SmallVector<int64_t> sourceShape;
  SmallVector<SliceAxis> axes;
};

// Pull the workgroup ptr out of a remapped source tile, checking the
// address space matches the LDS contract (any other address space means
// the caller has the wrong source and should fail the rewrite).
static FailureOr<std::pair<Value, PtrType>>
resolveWorkgroupPtr(Value remapped) {
  Value srcPtr = sourcePtr(remapped);
  if (!srcPtr)
    return failure();
  auto srcPtrType = dyn_cast<PtrType>(srcPtr.getType());
  if (!srcPtrType || srcPtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  return std::make_pair(srcPtr, srcPtrType);
}

// Resolve the view source for a buffer-view op: remap each per-axis
// subscript through the conversion pattern, then materialize the
// offset/stride axes the per-element loop walks.
static FailureOr<PtrViewSource>
resolveBufferViewSource(HCBufferViewOp bv,
                        ConversionPatternRewriter &rewriter) {
  Value sourceTile = bv.getBuffer();
  auto bareSrc = dyn_cast<BareTensorType>(sourceTile.getType());
  if (!bareSrc)
    return failure();
  FailureOr<SmallVector<int64_t>> sourceShape =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bareSrc));
  if (failed(sourceShape))
    return failure();
  Value remappedSrc = rewriter.getRemappedValue(sourceTile);
  if (!remappedSrc)
    return failure();
  FailureOr<std::pair<Value, PtrType>> ptr = resolveWorkgroupPtr(remappedSrc);
  if (failed(ptr))
    return failure();
  // Walk through remapped index operands so the per-axis offset/stride
  // values come out as `index`-typed SSA the per-element loop can plug
  // into `arith.muli`/`arith.addi` directly. The pre-conversion form
  // here is `!hc.idx<...>` / `!hc.slice<...>` from the buffer_view's
  // original operands, so we have to remap each index value (the slice
  // op's `lower`/`upper`/`step` are the slice op's own operands and get
  // remapped on its lowering, which has already happened by the time
  // any ptr-view consumer runs).
  SmallVector<Value> remappedIndices;
  remappedIndices.reserve(bv.getIndices().size());
  for (Value idx : bv.getIndices()) {
    Value remapped = rewriter.getRemappedValue(idx);
    if (!remapped)
      return failure();
    remappedIndices.push_back(remapped);
  }
  // Non-unit strides are fine here. `loadVectorFromPtrView` /
  // `writeVectorToWorkgroupPtr` walk the result-vector lanes through
  // `axis.stride * iter + axis.offset` (one per-element scalar
  // load/store), so any constant or symbolic stride lowers correctly
  // — there's no SIMD-vs-gather decision pending on this site.
  // The wider upgrade to `vector.gather` / strided `vector.transfer_*`
  // for performance is the separate "SIMD/gather upgrade" task.
  FailureOr<SmallVector<SliceAxis>> axes =
      collectAxes(bv.getOperation(), remappedIndices, rewriter,
                  /*requireUnitStride=*/false);
  if (failed(axes))
    return failure();
  if (axes->size() != sourceShape->size())
    return bv.emitOpError("expected one view subscript per source axis");
  return PtrViewSource{ptr->first, ptr->second, *sourceShape, std::move(*axes)};
}

// Synthesize a trivial axis pattern for non-view sources: each axis is
// a full slice (`offset = 0`, `stride = 1`) over the source's static
// dim. The strided per-element loop with trivial axes degenerates to
// the same flat `0..N` iteration the no-view path would emit.
static SmallVector<SliceAxis>
buildTrivialSliceAxes(ArrayRef<int64_t> shape,
                      ConversionPatternRewriter &rewriter, Location loc) {
  SmallVector<SliceAxis> trivialAxes;
  trivialAxes.reserve(shape.size());
  Value zero = zeroIndex(rewriter, loc);
  Value one = oneIndex(rewriter, loc);
  for (size_t i = 0; i < shape.size(); ++i) {
    SliceAxis ax;
    ax.offset = zero;
    ax.stride = one;
    ax.isSlice = true;
    trivialAxes.push_back(ax);
  }
  return trivialAxes;
}

// Walk back through a single `hc.buffer_view` to find the underlying
// workgroup ptr and the index pattern. The buffer_view ops sit in the
// pre-conversion IR — `getRemappedValue` walks `replaceOp` records to
// produce the post-conversion ptr for the source tile, which by this point
// has been lowered (or is being lowered in this same partial-conversion
// pass). Two-deep view chains (`view(view(...))`) are vanishingly rare in
// the surface; if they ever show up we'd unroll the chain here.
//
// For non-view sources (`hc.alloc`, fresh LDS from a `select`), we
// synthesize a trivial axis pattern (every axis a full slice over the
// source's static dim). That keeps the loader uniform — the strided
// per-element loop with trivial axes degenerates to the same flat
// `0..N` iteration the no-view path would emit.
static FailureOr<PtrViewSource>
resolvePtrViewSource(Value original, ConversionPatternRewriter &rewriter) {
  if (auto bv = original.getDefiningOp<HCBufferViewOp>())
    return resolveBufferViewSource(bv, rewriter);

  Value remapped = rewriter.getRemappedValue(original);
  if (!remapped)
    return failure();
  FailureOr<std::pair<Value, PtrType>> ptr = resolveWorkgroupPtr(remapped);
  if (failed(ptr))
    return failure();
  auto bare = dyn_cast<BareTensorType>(original.getType());
  if (!bare)
    return failure();
  FailureOr<SmallVector<int64_t>> shape =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bare));
  if (failed(shape))
    return failure();
  SmallVector<SliceAxis> trivialAxes =
      buildTrivialSliceAxes(*shape, rewriter, original.getLoc());
  return PtrViewSource{ptr->first, ptr->second, *shape, std::move(trivialAxes)};
}

// Per-element load of a multi-dim `vector<...xT>` from a workgroup ptr,
// honoring any `hc.buffer_view` index pattern in the source chain. Lane V
// reads from the source's flat offset `base + sum_k(view_idx[k] *
// source_stride[slice_axis_k])` — base is the contribution of scalar source
// axes, the slice contribution scales the lane coord by the source's
// identity-layout stride at that source axis. For trivial (full-slice)
// patterns, this collapses to the contiguous `0..N-1` linear walk a no-view
// source wants. For strided patterns (column-of-2D-tile in WMMA), it threads
// each lane through a separate `hc.ptr_offset` + `hc.ptr_load`.
//
// The per-element shape preserves the byte-per-element layout the
// cooperative store and other consumers use, sidestepping the i1
// packed/unpacked discrepancy LLVM has between scalar and vector i1 stores.
// Row-major stride for each axis of a static shape (last axis = 1,
// each preceding axis multiplied by the next axis's size).
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0; --axis)
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  return strides;
}

// Positions of slice (non-scalar) axes in `axes`, in source order.
static SmallVector<int64_t>
collectSliceAxisPositions(ArrayRef<SliceAxis> axes) {
  SmallVector<int64_t> positions;
  for (int64_t i = 0; i < static_cast<int64_t>(axes.size()); ++i)
    if (axes[i].isSlice)
      positions.push_back(i);
  return positions;
}

// Scalar (non-slice) source axes contribute a fixed base offset: each
// axis's `offset` value times its source-side row-major stride, summed
// into an `index`-typed running offset that's reused across every lane
// of the per-element load.
static Value computeScalarAxesBaseOffset(OpBuilder &rewriter, Location loc,
                                         ArrayRef<SliceAxis> axes,
                                         ArrayRef<int64_t> sourceStrides) {
  Value base = zeroIndex(rewriter, loc);
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i].isSlice)
      continue;
    Value strideVal =
        arith::ConstantIndexOp::create(rewriter, loc, sourceStrides[i])
            .getResult();
    Value contrib =
        arith::MulIOp::create(rewriter, loc, axes[i].offset, strideVal)
            .getResult();
    base = arith::AddIOp::create(rewriter, loc, base, contrib).getResult();
  }
  return base;
}

// Unflatten a linear lane index into per-axis view coords (row-major).
static SmallVector<int64_t> unflattenLaneIndex(int64_t lin,
                                               ArrayRef<int64_t> viewShape) {
  SmallVector<int64_t> coords(viewShape.size());
  int64_t remaining = lin;
  for (int64_t axis = static_cast<int64_t>(viewShape.size()) - 1; axis >= 0;
       --axis) {
    coords[axis] = remaining % viewShape[axis];
    remaining /= viewShape[axis];
  }
  return coords;
}

// Add per-lane slice-axis contributions to a running base offset. Each
// view coord scales the slice's stride, gets the slice's offset added,
// and then multiplies the source-side row-major stride at that axis.
static Value applyLaneSliceOffset(OpBuilder &rewriter, Location loc, Value base,
                                  ArrayRef<int64_t> coordInts,
                                  ArrayRef<int64_t> sliceAxisPositions,
                                  ArrayRef<SliceAxis> axes,
                                  ArrayRef<int64_t> sourceStrides) {
  Value flat = base;
  for (size_t vi = 0; vi < sliceAxisPositions.size(); ++vi) {
    int64_t srcAxisPos = sliceAxisPositions[vi];
    const SliceAxis &ax = axes[srcAxisPos];
    Value coordVal =
        arith::ConstantIndexOp::create(rewriter, loc, coordInts[vi])
            .getResult();
    Value scaledStride =
        arith::MulIOp::create(rewriter, loc, coordVal, ax.stride).getResult();
    Value indexInSource =
        arith::AddIOp::create(rewriter, loc, ax.offset, scaledStride)
            .getResult();
    Value srcStrideVal =
        arith::ConstantIndexOp::create(rewriter, loc, sourceStrides[srcAxisPos])
            .getResult();
    Value contrib =
        arith::MulIOp::create(rewriter, loc, indexInSource, srcStrideVal)
            .getResult();
    flat = arith::AddIOp::create(rewriter, loc, flat, contrib).getResult();
  }
  return flat;
}

static FailureOr<Value>
loadVectorFromPtrView(ConversionPatternRewriter &rewriter, Location loc,
                      Value original, mlir::VectorType vectorType,
                      ArrayRef<int64_t> viewShape) {
  FailureOr<PtrViewSource> src = resolvePtrViewSource(original, rewriter);
  if (failed(src))
    return failure();
  if (vectorType.getShape() != viewShape)
    return failure();

  SmallVector<int64_t> sourceStrides = computeRowMajorStrides(src->sourceShape);
  SmallVector<int64_t> sliceAxisPositions =
      collectSliceAxisPositions(src->axes);
  if (sliceAxisPositions.size() != viewShape.size())
    return failure();

  Value baseOffset =
      computeScalarAxesBaseOffset(rewriter, loc, src->axes, sourceStrides);

  int64_t total = 1;
  for (int64_t d : viewShape)
    total *= d;
  Type elementType = vectorType.getElementType();
  Value result = arith::ConstantOp::create(rewriter, loc, vectorType,
                                           rewriter.getZeroAttr(vectorType))
                     .getResult();
  for (int64_t lin = 0; lin != total; ++lin) {
    SmallVector<int64_t> coordInts = unflattenLaneIndex(lin, viewShape);
    Value flat =
        applyLaneSliceOffset(rewriter, loc, baseOffset, coordInts,
                             sliceAxisPositions, src->axes, sourceStrides);
    Value addr =
        HCPtrOffsetOp::create(rewriter, loc, src->ptrType, src->sourcePtr, flat)
            .getResult();
    Value element =
        HCPtrLoadOp::create(rewriter, loc, elementType, addr).getResult();
    result = vector::InsertOp::create(rewriter, loc, element, result, coordInts)
                 .getResult();
  }
  return result;
}

// Materialize a vector value as the converter's chosen carrier:
//   * `vector<...xT>` is the per-workitem register path — pass through.
//   * `!hc.ptr<workgroup, T>` is the LDS-staged path — allocate the tile
//     and write the lanes per-element.
//
// Semantic `!hc.tensor` doesn't appear here by contract:
// `hc-decompose-shaped-values` splits every tensor producer/consumer
// into bare (data, mask) pairs upstream so launch-body only sees the
// bare carrier. Anything else surviving to this point is a contract
// bug in decompose.
//
// Allocate an LDS tile sized for `shape`'s element count and write
// the per-lane `vector` into it.
// Pull the scalar splat element out of a constant vector. Returns a
// fresh scalar `arith.constant` SSA value matching the vector's element
// type when `vector` is a splat constant, or `nullptr` otherwise. Every
// current caller of `writeVectorToFreshLDS` (zeros / ones / full /
// full_mask / broadcast-of-scalar fill) feeds in a splat, so this
// captures the common case; a non-splat vector means the caller already
// has per-slot values and falls through to the unrolled-store path.
static Value extractSplatScalar(OpBuilder &builder, Location loc,
                                Value vector) {
  auto cst = vector.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return Value{};
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return Value{};
  TypedAttr scalarAttr = dense.getSplatValue<TypedAttr>();
  return arith::ConstantOp::create(builder, loc, scalarAttr).getResult();
}

// Emit a workgroup-cooperative fill of a freshly allocated LDS tile as a
// single `hc.generic` with one parallel iter per slot. `hc-lower-generic`
// then routes this through `lowerCollective`, which dispatches one slot
// per thread. Net effect:
//
//   * fill cost drops from `wgSize * total` per-thread stores to one
//     pass over the slot range divided across the workgroup;
//   * every workgroup-AS write lives inside a structured `hc.generic`,
//     which keeps the surface uniform for `hc-insert-workgroup-barriers`
//     — the dedicated pass that owns cross-generic synchronization on
//     workgroup-AS storage.
//
// `lin` is the canonical name for the flat slot iter sym; ixsimpl
// hash-conses the `#hc.expr<"lin">` payload across uses so the canonical
// handle stays cheap to compare downstream.
static LogicalResult emitInitFillGeneric(OpBuilder &builder, Location loc,
                                         Value lds, PtrType ptrType,
                                         Value scalar, int64_t total) {
  MLIRContext *ctx = builder.getContext();
  Type elemTy = ptrType.getElementType();
  if (!elemTy || scalar.getType() != elemTy)
    return failure();

  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  FailureOr<sym::ExprHandle> linExpr = sym::composeExprSym(store, "lin");
  if (failed(linExpr))
    return failure();
  ArrayAttr outOff = ArrayAttr::get(ctx, {ExprAttr::get(ctx, *linExpr)});
  ArrayAttr insOffsets = ArrayAttr::get(ctx, {});
  ArrayAttr outsOffsets = ArrayAttr::get(ctx, {outOff});

  StringAttr linSym = StringAttr::get(ctx, "lin");
  ArrayAttr iterSyms = ArrayAttr::get(ctx, {linSym});
  ArrayAttr iterKinds =
      ArrayAttr::get(ctx, {IterKindAttr::get(ctx, IterKind::Parallel)});

  Value totalVal =
      arith::ConstantIndexOp::create(builder, loc, total).getResult();

  auto generic = HCGenericOp::create(
      builder, loc,
      /*resultTypes=*/TypeRange{}, iterSyms, ValueRange(totalVal), iterKinds,
      /*ins=*/ValueRange{}, /*outs=*/ValueRange{lds},
      /*ambient_idxs=*/ValueRange{},
      /*ambient_idx_syms=*/ArrayAttr::get(ctx, {}), insOffsets, outsOffsets);

  Block *body = new Block();
  body->addArgument(elemTy, loc);
  generic.getBody().push_back(body);
  OpBuilder bodyBuilder(body, body->begin());
  HCYieldOp::create(bodyBuilder, loc, ValueRange{scalar});
  return success();
}

static FailureOr<Value> writeVectorToFreshLDS(OpBuilder &builder, Location loc,
                                              PtrType ptrType, Value vector,
                                              ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t d : shape)
    total *= d;
  Value lds = allocateWorkgroupPtr(builder, loc, ptrType, total);

  // Splat path: lift the fill into a structured `hc.generic` so the
  // workgroup-cooperative lowering (and the future barrier-insertion
  // pass) sees a uniform surface. Captures every caller in tree today.
  if (Value scalar = extractSplatScalar(builder, loc, vector)) {
    if (failed(emitInitFillGeneric(builder, loc, lds, ptrType, scalar, total)))
      return failure();
    return lds;
  }

  // Fallback for the (currently unused) non-splat path: unrolled
  // per-element stores. Leaves the stray-store pattern outside the
  // generic-only contract; if a real caller appears, route it through
  // a per-slot `hc.generic` instead so the barrier pass keeps working
  // without special-casing this site.
  if (failed(writeVectorToWorkgroupPtr(builder, loc, vector, lds, shape)))
    return failure();
  return lds;
}

static FailureOr<Value>
materializeShapedResult(OpBuilder &builder, Location loc, Type convertedType,
                        Value vector, ArrayRef<int64_t> shape) {
  if (auto vectorType = dyn_cast_if_present<mlir::VectorType>(convertedType)) {
    if (vector.getType() != vectorType)
      return failure();
    return vector;
  }
  if (auto ptrType = dyn_cast_if_present<PtrType>(convertedType)) {
    if (ptrType.getAddrSpace() != AddrSpace::Workgroup)
      return failure();
    return writeVectorToFreshLDS(builder, loc, ptrType, vector, shape);
  }
  return failure();
}

// Read a converted shaped value as a multi-dim vector. The per-lane path
// is a no-op (the value is already a vector); the LDS path expands into
// per-element `hc.ptr_load`s, walking back through any `hc.buffer_view`
// chain on the original source so strided views (column-of-2D, etc.)
// produce correctly strided per-lane offsets.
static FailureOr<Value> shapedValueAsVector(ConversionPatternRewriter &rewriter,
                                            Location loc, Value original,
                                            Value remapped, Type convertedType,
                                            ArrayRef<int64_t> shape) {
  if (auto vectorType = dyn_cast<mlir::VectorType>(convertedType)) {
    if (remapped.getType() == vectorType)
      return remapped;
    return failure();
  }
  auto ptrType = dyn_cast<PtrType>(convertedType);
  if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  Type elementType = ptrType.getElementType();
  if (!elementType)
    return failure();
  mlir::VectorType vectorType = mlir::VectorType::get(shape, elementType);
  return loadVectorFromPtrView(rewriter, loc, original, vectorType, shape);
}

static Value extractVectorElement(OpBuilder &builder, Location loc,
                                  Value vector, ArrayRef<int64_t> coordinates) {
  if (coordinates.empty())
    return vector;
  SmallVector<OpFoldResult> positions;
  positions.reserve(coordinates.size());
  for (int64_t coordinate : coordinates)
    positions.push_back(builder.getI64IntegerAttr(coordinate));
  return vector::ExtractOp::create(builder, loc, vector, positions).getResult();
}

static SmallVector<SmallVector<int64_t>>
staticVectorCoordinates(ArrayRef<int64_t> shape) {
  int64_t elementCount = 1;
  for (int64_t dim : shape)
    elementCount *= dim;

  SmallVector<SmallVector<int64_t>> coordinates;
  coordinates.reserve(elementCount);
  for (int64_t linear = 0; linear != elementCount; ++linear) {
    int64_t remaining = linear;
    SmallVector<int64_t> coordinate(shape.size(), 0);
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0;
         --axis) {
      coordinate[axis] = remaining % shape[axis];
      remaining /= shape[axis];
    }
    coordinates.push_back(std::move(coordinate));
  }
  return coordinates;
}

static Value scaleIndexOffset(OpBuilder &builder, Location loc, Value stride,
                              int64_t coordinate) {
  if (coordinate == 0)
    return arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  if (coordinate == 1)
    return stride;
  Value coordinateValue =
      arith::ConstantIndexOp::create(builder, loc, coordinate);
  return arith::MulIOp::create(builder, loc, stride, coordinateValue)
      .getResult();
}

static SmallVector<Value> storeIndicesForCoordinate(OpBuilder &builder,
                                                    Location loc,
                                                    ArrayRef<SliceAxis> axes,
                                                    ArrayRef<int64_t> coords) {
  SmallVector<Value> indices;
  indices.reserve(axes.size());
  int64_t sliceAxis = 0;
  for (const SliceAxis &axis : axes) {
    if (!axis.isSlice) {
      indices.push_back(axis.offset);
      continue;
    }
    Value scaled =
        scaleIndexOffset(builder, loc, axis.stride, coords[sliceAxis++]);
    indices.push_back(
        arith::AddIOp::create(builder, loc, axis.offset, scaled).getResult());
  }
  return indices;
}

// Build the per-axis index list for one lane of the result vector. The
// caller feeds these into `hc.ptr_offset` + `hc.ptr_load` against the
// kernel-arg pointer. `axes` carries the slice's per-axis offsets and
// strides (`offset + coord * stride` for slice axes; just `offset` for
// scalar axes). The result vector's coords identify which slice axis we're
// walking; non-slice axes get the same `axes[k].offset` for every lane.
static SmallVector<Value> kernelArgLaneIndices(OpBuilder &builder, Location loc,
                                               ArrayRef<SliceAxis> axes,
                                               ArrayRef<int64_t> resultCoord) {
  SmallVector<Value> indices;
  indices.reserve(axes.size());
  int64_t sliceCursor = 0;
  for (const SliceAxis &info : axes) {
    if (!info.isSlice) {
      indices.push_back(info.offset);
      continue;
    }
    int64_t coord = resultCoord[sliceCursor++];
    Value scaled = scaleIndexOffset(builder, loc, info.stride, coord);
    indices.push_back(
        arith::AddIOp::create(builder, loc, info.offset, scaled).getResult());
  }
  return indices;
}

// Result shape/type bundle for a load-like op: the converted vector
// type, its static shape, and the element type.
struct LoadLikeResultShape {
  mlir::VectorType vectorType;
  SmallVector<int64_t> shape;
  Type elementType;
};

// Decode the load's bare-vector result into a static `(VectorType,
// shape, elementType)` bundle. Returns failure when the converter
// wouldn't produce a vector or the bare-vector source isn't statically
// shaped — both indicate the pattern shouldn't fire.
static FailureOr<LoadLikeResultShape>
unpackLoadLikeResultShape(const TypeConverter &converter, Type bareResultType) {
  Type converted = converter.convertType(bareResultType);
  auto vectorType = dyn_cast_if_present<mlir::VectorType>(converted);
  if (!vectorType)
    return failure();
  auto bareVector = dyn_cast<BareVectorType>(bareResultType);
  if (!bareVector)
    return failure();
  FailureOr<SmallVector<int64_t>> dims =
      staticIntegerShape(cast<SymbolicallyShapedTypeInterface>(bareVector));
  if (failed(dims))
    return failure();
  return LoadLikeResultShape{vectorType, std::move(*dims),
                             vectorType.getElementType()};
}

// Resolve the per-axis slice/scalar axes for a kernel-arg-backed
// access (load or store). The post-flatten synthesizer handles the
// rank-1 carrier case; pre-flatten ops fall back to the generic axis
// collector.
static FailureOr<SmallVector<SliceAxis>>
resolveAccessAxes(Operation *op, ValueRange indices,
                  ConversionPatternRewriter &rewriter,
                  const KernelArgSource &kernelArg, ArrayRef<int64_t> shape) {
  if (auto synthesized = synthesizePostFlattenAxes(rewriter, op->getLoc(),
                                                   kernelArg, indices, shape))
    return std::move(*synthesized);
  return collectAxes(op, indices, rewriter, /*requireUnitStride=*/false);
}

// Emit the per-lane scalar load chain for a kernel-arg-backed load:
// one `hc.ptr_offset` + `hc.ptr_load` + `vector.insert` per result
// coord, threaded through a running vector accumulator initialized to
// zero of the result vector type. LLVM's SLP recombines adjacent
// scalar loads when the slice is unit-stride.
static Value emitLoadLikeLanes(OpBuilder &rewriter, Location loc,
                               const KernelArgSource &kernelArg,
                               PtrType sourcePtrType,
                               const LoadLikeResultShape &result,
                               ArrayRef<SliceAxis> axes) {
  Value laneVec =
      arith::ConstantOp::create(rewriter, loc, result.vectorType,
                                rewriter.getZeroAttr(result.vectorType))
          .getResult();
  for (ArrayRef<int64_t> resultCoord : staticVectorCoordinates(result.shape)) {
    SmallVector<Value> indices =
        kernelArgLaneIndices(rewriter, loc, axes, resultCoord);
    Value flat = linearizeKernelArgOffset(rewriter, loc, kernelArg, indices);
    Value addr =
        HCPtrOffsetOp::create(rewriter, loc, sourcePtrType, kernelArg.ptr, flat)
            .getResult();
    Value elem = HCPtrLoadOp::create(rewriter, loc, result.elementType, addr)
                     .getResult();
    laneVec =
        vector::InsertOp::create(rewriter, loc, elem, laneVec, resultCoord)
            .getResult();
  }
  return laneVec;
}

// Per-lane vector load from a kernel-arg `!hc.ptr<global, T>`. Every lane
// materializes its own fragment via per-element scalar `hc.ptr_offset` +
// `hc.ptr_load` + `vector.insert`s; LLVM's SLP recombines adjacent
// scalar loads when the slice is unit-stride. Switching to per-element
// keeps a single shape for unit and non-unit stride and sidesteps the
// i1 packed-vs-byte discrepancy that `vector.transfer_read` of
// `vector<Nxi1>` triggered.
//
// `hc.load` / `hc.vload` reaching launch-body always have bare-vector
// results: bare-tensor loads (workgroup-shared tiles) are funneled
// through `hc-load-store-to-generic` + `hc-flatten-with-layouts` +
// `hc-lower-generic` long before this pass walks them.
template <typename OpT>
struct ConvertLoadLikeOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LoadLikeResultShape> result = unpackLoadLikeResultShape(
        *this->typeConverter, op.getResult().getType());
    if (failed(result))
      return failure();

    Value source = [&]() -> Value {
      if constexpr (std::is_same_v<OpT, HCLoadOp>)
        return adaptor.getBuffer();
      else
        return adaptor.getSource();
    }();
    std::optional<KernelArgSource> kernelArg =
        resolveAccessKernelArg(rewriter, op.getLoc(), source);
    if (!kernelArg)
      return op.emitOpError(
          "expected load source to be a kernel-arg ptr ABI cast");
    if (kernelArg->rank() != static_cast<unsigned>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected kernel-arg source rank to match index rank");
    auto sourcePtrType = cast<PtrType>(kernelArg->ptr.getType());

    FailureOr<SmallVector<SliceAxis>> axes =
        resolveAccessAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                          *kernelArg, result->shape);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != static_cast<int64_t>(result->shape.size()))
      return op.emitOpError("load result rank must match slice subscript rank");

    Value laneVec = emitLoadLikeLanes(rewriter, op.getLoc(), *kernelArg,
                                      sourcePtrType, *result, *axes);
    rewriter.replaceOp(op, laneVec);
    return success();
  }
};

// Static shape from any symbolically shaped result type. Drives per-element
// materialization on every flavour the launch-body lowering sees: BareTensor
// (LDS-staged), BareVector (per-workitem vector register), and the semantic
// Tensor / Vector carriers that `hc-decompose-shaped-values --strict=false`
// leaves behind on ops it doesn't decompose. The semantic types ride through
// the converter as identity; the rewrite still needs a literal int shape to
// build the vector init `arith.constant` carrier off of.
static FailureOr<SmallVector<int64_t>> shapedResultShape(Type type) {
  if (auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type))
    return staticIntegerShape(shaped);
  return failure();
}

struct ConvertFullMaskOp : public OpConversionPattern<HCFullMaskOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCFullMaskOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getMask().getType());
    if (!converted)
      return failure();

    FailureOr<SmallVector<int64_t>> shape =
        shapedResultShape(op.getMask().getType());
    if (succeeded(shape)) {
      mlir::VectorType vectorType =
          mlir::VectorType::get(*shape, rewriter.getI1Type());
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, 1);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      FailureOr<Value> result = materializeShapedResult(
          rewriter, op.getLoc(), converted, vector, *shape);
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
      return success();
    }
    FailureOr<TypedAttr> attr = splatAttr(rewriter, converted, 1);
    if (failed(attr))
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, converted, *attr);
    return success();
  }
};

template <typename OpT, int64_t FillValue>
struct ConvertNullaryShapedConstantOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();

    FailureOr<SmallVector<int64_t>> shape =
        shapedResultShape(op.getResult().getType());
    if (succeeded(shape)) {
      auto origShaped =
          cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
      Type elementType =
          convertElementType(origShaped.getSymbolicElementType());
      if (!elementType)
        return failure();
      mlir::VectorType vectorType = mlir::VectorType::get(*shape, elementType);
      FailureOr<TypedAttr> attr = splatAttr(rewriter, vectorType, FillValue);
      if (failed(attr))
        return failure();
      Value vector =
          arith::ConstantOp::create(rewriter, op.getLoc(), vectorType, *attr);
      FailureOr<Value> result = materializeShapedResult(
          rewriter, op.getLoc(), converted, vector, *shape);
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
      return success();
    }
    FailureOr<TypedAttr> attr = splatAttr(rewriter, converted, FillValue);
    if (failed(attr))
      return failure();
    rewriter.template replaceOpWithNewOp<arith::ConstantOp>(op, converted,
                                                            *attr);
    return success();
  }
};

template <typename OpT>
struct ConvertFillShapedConstantOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    FailureOr<SmallVector<int64_t>> shape =
        shapedResultShape(op.getResult().getType());
    if (failed(shape))
      return failure();
    auto origShaped =
        cast<SymbolicallyShapedTypeInterface>(op.getResult().getType());
    Type elementType = convertElementType(origShaped.getSymbolicElementType());
    if (!elementType)
      return failure();
    mlir::VectorType vectorType = mlir::VectorType::get(*shape, elementType);
    Value vector = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                               vectorType, adaptor.getValue());
    FailureOr<Value> result = materializeShapedResult(
        rewriter, op.getLoc(), converted, vector, *shape);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

struct ConvertEmptyOp : public OpConversionPattern<HCEmptyOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCEmptyOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = dyn_cast_if_present<PtrType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!ptrType || ptrType.getAddrSpace() != AddrSpace::Workgroup)
      return failure();
    auto bare = dyn_cast<BareTensorType>(op.getResult().getType());
    if (!bare)
      return failure();
    FailureOr<int64_t> count = bareTensorElementCount(bare);
    if (failed(count))
      return failure();
    rewriter.replaceOp(
        op, allocateWorkgroupPtr(rewriter, op.getLoc(), ptrType, *count));
    return success();
  }
};

struct ConvertVecOp : public OpConversionPattern<HCVecOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCVecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    auto vectorType = dyn_cast_if_present<mlir::VectorType>(converted);
    if (!vectorType)
      return failure();

    // bare_tensor → bare_vector: the bare_tensor source converts to a
    // workgroup ptr; load it into the per-lane vector via per-element
    // `hc.ptr_load`s, walking back through any `hc.buffer_view` chain so
    // strided views (e.g. column of a 2D tile) get correctly strided
    // per-lane offsets.
    if (auto ptrType = dyn_cast<PtrType>(adaptor.getValue().getType())) {
      if (ptrType.getAddrSpace() != AddrSpace::Workgroup)
        return failure();
      FailureOr<SmallVector<int64_t>> shape =
          shapedResultShape(op.getValue().getType());
      if (failed(shape))
        return failure();
      FailureOr<Value> vector = loadVectorFromPtrView(
          rewriter, op.getLoc(), op.getValue(), vectorType, *shape);
      if (failed(vector))
        return failure();
      rewriter.replaceOp(op, *vector);
      return success();
    }
    if (!converted || converted != adaptor.getValue().getType())
      return failure();
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<HCSelectOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();

    // Bare-tensor result: condition + true value are workgroup ptrs, load
    // them as vectors, blend, store the result back to a fresh LDS tile.
    if (auto ptrType = dyn_cast<PtrType>(converted)) {
      if (ptrType.getAddrSpace() != AddrSpace::Workgroup)
        return failure();
      FailureOr<SmallVector<int64_t>> shape =
          shapedResultShape(op.getResult().getType());
      if (failed(shape))
        return failure();
      Type elementType = ptrType.getElementType();
      mlir::VectorType vectorType = mlir::VectorType::get(*shape, elementType);
      mlir::VectorType maskType =
          mlir::VectorType::get(*shape, rewriter.getI1Type());

      FailureOr<Value> condition = loadVectorFromPtrView(
          rewriter, op.getLoc(), op.getCondition(), maskType, *shape);
      FailureOr<Value> trueValue = loadVectorFromPtrView(
          rewriter, op.getLoc(), op.getTrueValue(), vectorType, *shape);
      if (failed(condition) || failed(trueValue))
        return op.emitOpError(
            "expected tensor select operands to lower to workgroup ptrs");

      Value falseValue =
          vector::BroadcastOp::create(rewriter, op.getLoc(), vectorType,
                                      adaptor.getFalseValue())
              .getResult();
      Value selected =
          arith::SelectOp::create(rewriter, op.getLoc(), vectorType, *condition,
                                  *trueValue, falseValue)
              .getResult();
      FailureOr<Value> result = materializeShapedResult(
          rewriter, op.getLoc(), converted, selected, *shape);
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
      return success();
    }

    Value falseValue = adaptor.getFalseValue();
    if (isa<mlir::VectorType>(converted))
      falseValue = vector::BroadcastOp::create(rewriter, op.getLoc(), converted,
                                               falseValue)
                       .getResult();

    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, converted, adaptor.getCondition(), adaptor.getTrueValue(),
        falseValue);
    return success();
  }
};

// Lower the store source into a vector value plus its static shape,
// running through the same `shapedValueAsVector` path that load
// patterns use. Diagnoses both the static-shape failure and the
// non-vector lowering on `op`.
static FailureOr<std::pair<Value, mlir::VectorType>>
lowerStoreSource(HCStoreOp op, HCStoreOp::Adaptor adaptor,
                 ConversionPatternRewriter &rewriter,
                 const TypeConverter &converter) {
  Type convertedSource = converter.convertType(op.getSource().getType());
  FailureOr<SmallVector<int64_t>> sourceShape =
      shapedResultShape(op.getSource().getType());
  if (failed(sourceShape))
    return failure();
  FailureOr<Value> source =
      shapedValueAsVector(rewriter, op.getLoc(), op.getSource(),
                          adaptor.getSource(), convertedSource, *sourceShape);
  if (failed(source))
    return op.emitOpError("expected store source to lower to a vector");
  auto sourceType = dyn_cast<mlir::VectorType>(source->getType());
  if (!sourceType)
    return op.emitOpError("expected store source to be a vector");
  return std::make_pair(*source, sourceType);
}

// Lower the optional store mask into an `i1` vector of the same shape
// as `sourceType`. Returns a null Value when the op has no mask. Any
// shape / type mismatch on the lowered mask is reported on `op`.
static FailureOr<Value> lowerStoreMask(HCStoreOp op, HCStoreOp::Adaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       const TypeConverter &converter,
                                       mlir::VectorType sourceType) {
  Value originalMask = op.getMask();
  if (!originalMask)
    return Value{};
  Type convertedMask = converter.convertType(originalMask.getType());
  FailureOr<SmallVector<int64_t>> maskShape =
      shapedResultShape(originalMask.getType());
  if (failed(maskShape))
    return failure();
  FailureOr<Value> maskVector =
      shapedValueAsVector(rewriter, op.getLoc(), originalMask,
                          adaptor.getMask(), convertedMask, *maskShape);
  if (failed(maskVector))
    return op.emitOpError("expected store mask to lower to a vector");
  auto maskType = dyn_cast<mlir::VectorType>(maskVector->getType());
  auto expectedMaskType =
      mlir::VectorType::get(sourceType.getShape(), rewriter.getI1Type());
  if (maskType != expectedMaskType)
    return op.emitOpError("store mask type ")
           << maskVector->getType() << " must match " << expectedMaskType;
  return *maskVector;
}

// Emit per-lane scalar stores for `source` into the kernel-arg buffer.
// `mask` may be null (unconditional stores via `hc.ptr_store`); when
// present, each lane's mask bit guards a `hc.ptr_store_pred` so the
// mask rides as a first-class operand instead of via an `scf.if`,
// matching the symmetric `hc.ptr_load_pred` emitted for masked loads.
static void emitStoreLanes(OpBuilder &rewriter, Location loc,
                           const KernelArgSource &kernelArg,
                           PtrType destPtrType, Value source,
                           mlir::VectorType sourceType, Value mask,
                           ArrayRef<SliceAxis> axes) {
  for (ArrayRef<int64_t> coordinate :
       staticVectorCoordinates(sourceType.getShape())) {
    Value element = extractVectorElement(rewriter, loc, source, coordinate);
    SmallVector<Value> indices =
        storeIndicesForCoordinate(rewriter, loc, axes, coordinate);
    Value flat = linearizeKernelArgOffset(rewriter, loc, kernelArg, indices);
    Value addr =
        HCPtrOffsetOp::create(rewriter, loc, destPtrType, kernelArg.ptr, flat)
            .getResult();
    if (!mask) {
      HCPtrStoreOp::create(rewriter, loc, element, addr);
      continue;
    }
    Value guard = extractVectorElement(rewriter, loc, mask, coordinate);
    HCPtrStorePredOp::create(rewriter, loc, element, addr, guard);
  }
}

struct ConvertStoreOp : public OpConversionPattern<HCStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<KernelArgSource> kernelArg =
        resolveAccessKernelArg(rewriter, op.getLoc(), adaptor.getDest());
    if (!kernelArg)
      return op.emitOpError(
          "expected store destination to be a kernel-arg ptr ABI cast");
    if (kernelArg->rank() != static_cast<unsigned>(adaptor.getIndices().size()))
      return op.emitOpError(
          "expected kernel-arg destination rank to match index rank");
    auto destPtrType = cast<PtrType>(kernelArg->ptr.getType());

    FailureOr<std::pair<Value, mlir::VectorType>> source =
        lowerStoreSource(op, adaptor, rewriter, *typeConverter);
    if (failed(source))
      return failure();
    auto [sourceValue, sourceType] = *source;
    SmallVector<int64_t> sourceShape(sourceType.getShape().begin(),
                                     sourceType.getShape().end());

    FailureOr<SmallVector<SliceAxis>> axes =
        resolveAccessAxes(op.getOperation(), adaptor.getIndices(), rewriter,
                          *kernelArg, sourceShape);
    if (failed(axes))
      return failure();
    if (llvm::count_if(*axes, [](const SliceAxis &axis) {
          return axis.isSlice;
        }) != sourceType.getRank())
      return op.emitOpError(
          "store source rank must match slice subscript rank");

    FailureOr<Value> mask =
        lowerStoreMask(op, adaptor, rewriter, *typeConverter, sourceType);
    if (failed(mask))
      return failure();

    emitStoreLanes(rewriter, op.getLoc(), *kernelArg, destPtrType, sourceValue,
                   sourceType, *mask, *axes);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCallIntrinsicOp : public OpConversionPattern<HCCallIntrinsicOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCCallIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    operands.reserve(op.getArgs().size());
    for (auto [original, converted] :
         llvm::zip_equal(op.getArgs(), adaptor.getArgs())) {
      Type boundary =
          convertIntrinsicBoundaryType(original.getType(), *typeConverter);
      if (!boundary)
        return failure();
      operands.push_back(
          castIfNeeded(rewriter, op.getLoc(), converted, boundary));
    }

    SmallVector<Type> results;
    results.reserve(op.getResultTypes().size());
    for (Type result : op.getResultTypes()) {
      Type boundary = convertIntrinsicBoundaryType(result, *typeConverter);
      if (!boundary)
        return failure();
      results.push_back(boundary);
    }

    OperationState state(op.getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(results);
    state.addAttributes(op->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

// bare_tensor → bare_tensor view: source is a workgroup ptr. The
// strided per-element loader (`loadVectorFromPtrView`) walks back
// through this op to reconstruct the source-axis pattern, so the
// buffer_view itself just needs to type-resolve cleanly. We pass the
// source ptr through unchanged; the op survives in the IR as a
// metadata anchor for the consumer chain walk and gets DCE'd once
// every consumer has lowered. (The result-ptr type matches the source
// because both are workgroup-AS, same element type, rank-erased.)
static LogicalResult
lowerWorkgroupBufferView(HCBufferViewOp op, HCBufferViewOp::Adaptor adaptor,
                         ConversionPatternRewriter &rewriter, Type converted,
                         PtrType sourcePtrType) {
  if (sourcePtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  auto resultPtrType = dyn_cast_if_present<PtrType>(converted);
  if (!resultPtrType || resultPtrType.getAddrSpace() != AddrSpace::Workgroup)
    return failure();
  if (resultPtrType.getElementType() != sourcePtrType.getElementType())
    return failure();
  rewriter.replaceOp(op, adaptor.getBuffer());
  return success();
}

// Partition `localAxes` into (a) the permutation that pulls scalar
// (non-slice) axes to the front (in original axis order) followed by
// the slice axes in original order, and (b) the per-scalar `offset`
// values that the downstream `vector.extract` uses as positions.
static void
buildVectorViewPermutation(ArrayRef<SliceAxis> localAxes,
                           SmallVectorImpl<int64_t> &permutation,
                           SmallVectorImpl<OpFoldResult> &positions) {
  permutation.reserve(localAxes.size());
  for (auto [axis, info] : llvm::enumerate(localAxes)) {
    if (info.isSlice)
      continue;
    permutation.push_back(axis);
    positions.push_back(info.offset);
  }
  for (auto [axis, info] : llvm::enumerate(localAxes)) {
    if (info.isSlice)
      permutation.push_back(axis);
  }
}

// Reshape `result` to match `converted` exactly. Both sides must be
// vectors; otherwise the conversion isn't expressible and we bail.
static FailureOr<Value> reshapeToConvertedVector(OpBuilder &rewriter,
                                                 Location loc, Value result,
                                                 Type converted) {
  if (result.getType() == converted)
    return result;
  if (!isa<mlir::VectorType>(result.getType()) ||
      !isa<mlir::VectorType>(converted))
    return failure();
  return vector::ShapeCastOp::create(rewriter, loc, converted, result)
      .getResult();
}

// Vector-shape view: lower an `hc.buffer_view` over a vector source
// into `vector.transpose` (scalar axes to the front) + `vector.extract`
// (drop the scalar prefix coordinates) + a final shape cast back to
// the converter's expected result type.
static LogicalResult lowerVectorBufferView(HCBufferViewOp op,
                                           HCBufferViewOp::Adaptor adaptor,
                                           ConversionPatternRewriter &rewriter,
                                           Type converted) {
  auto sourceType = dyn_cast<mlir::VectorType>(adaptor.getBuffer().getType());
  if (!sourceType || !converted)
    return failure();

  FailureOr<SmallVector<SliceAxis>> axes =
      collectAxes(op.getOperation(), adaptor.getIndices(), rewriter);
  if (failed(axes))
    return failure();
  if (static_cast<int64_t>(axes->size()) < sourceType.getRank())
    return op.emitOpError(
        "expected at least one view subscript per vector axis");

  ArrayRef<SliceAxis> localAxes(*axes);
  localAxes = localAxes.take_front(sourceType.getRank());
  SmallVector<int64_t> permutation;
  SmallVector<OpFoldResult> positions;
  buildVectorViewPermutation(localAxes, permutation, positions);

  Value source = adaptor.getBuffer();
  if (!llvm::equal(permutation, llvm::seq<int64_t>(0, localAxes.size())))
    source =
        vector::TransposeOp::create(rewriter, op.getLoc(), source, permutation)
            .getResult();

  Value result = source;
  if (!positions.empty())
    result = vector::ExtractOp::create(rewriter, op.getLoc(), source, positions)
                 .getResult();

  FailureOr<Value> reshaped =
      reshapeToConvertedVector(rewriter, op.getLoc(), result, converted);
  if (failed(reshaped))
    return failure();
  rewriter.replaceOp(op, *reshaped);
  return success();
}

struct ConvertBufferViewOp : public OpConversionPattern<HCBufferViewOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBufferViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (auto srcPtr = dyn_cast<PtrType>(adaptor.getBuffer().getType()))
      return lowerWorkgroupBufferView(op, adaptor, rewriter, converted, srcPtr);
    return lowerVectorBufferView(op, adaptor, rewriter, converted);
  }
};

struct ConvertDivOp : public OpConversionPattern<HCDivOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (converted.isIntOrIndex()) {
      rewriter.replaceOpWithNewOp<arith::DivUIOp>(
          op, converted, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    if (isa<FloatType>(converted)) {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(
          op, converted, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    return failure();
  }
};

struct ConvertNegOp : public OpConversionPattern<HCNegOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCNegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (converted.isIntOrIndex()) {
      Value zero =
          converted.isIndex()
              ? arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0)
                    .getResult()
              : arith::ConstantOp::create(rewriter, op.getLoc(), converted,
                                          rewriter.getZeroAttr(converted));
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, converted, zero,
                                                 adaptor.getValue());
      return success();
    }
    if (isa<FloatType>(converted)) {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, converted,
                                                 adaptor.getValue());
      return success();
    }
    return failure();
  }
};

template <typename OpT> static constexpr arith::CmpIPredicate cmpPredicate() {
  if constexpr (std::is_same_v<OpT, HCCmpLtOp>)
    return arith::CmpIPredicate::slt;
  if constexpr (std::is_same_v<OpT, HCCmpLeOp>)
    return arith::CmpIPredicate::sle;
  if constexpr (std::is_same_v<OpT, HCCmpGtOp>)
    return arith::CmpIPredicate::sgt;
  if constexpr (std::is_same_v<OpT, HCCmpGeOp>)
    return arith::CmpIPredicate::sge;
  if constexpr (std::is_same_v<OpT, HCCmpEqOp>)
    return arith::CmpIPredicate::eq;
  if constexpr (std::is_same_v<OpT, HCCmpNeOp>)
    return arith::CmpIPredicate::ne;
}

template <typename OpT> struct ConvertCmpOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = this->typeConverter->convertType(op.getResult().getType());
    if (!converted || !converted.isInteger(1))
      return failure();
    rewriter.template replaceOpWithNewOp<arith::CmpIOp>(
        op, cmpPredicate<OpT>(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertCastOp : public OpConversionPattern<HCCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted)
      return failure();
    if (adaptor.getSource().getType() == converted) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, converted, adaptor.getSource());
    return success();
  }
};

struct ConvertBufferDimOp : public OpConversionPattern<HCBufferDimOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCBufferDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type converted = typeConverter->convertType(op.getDim().getType());
    if (!converted || !converted.isIndex())
      return failure();
    std::optional<KernelArgSource> kernelArg =
        resolveKernelArg(adaptor.getBuffer());
    if (!kernelArg)
      return op.emitOpError("expected buffer to be a kernel-arg ptr ABI cast");
    int64_t axis = op.getAxis();
    if (axis < 0 || static_cast<unsigned>(axis) >= kernelArg->rank())
      return op.emitOpError(
          "buffer_dim axis out of range for the kernel-arg source");
    rewriter.replaceOp(op, kernelArg->dims[axis]);
    return success();
  }
};

template <typename OpT>
struct AdaptRegionlessOp : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> results;
    if (failed(
            this->typeConverter->convertTypes(op->getResultTypes(), results)))
      return failure();
    OperationState state(op.getLoc(), op->getName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(results);
    state.addAttributes(op->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

// Resolve each `!hc.buffer<T, [...]>` operand in `operands` back to
// its underlying `!hc.ptr<global, T>` via the kernel-arg UCC chain.
// Non-buffer operands are untouched; `changed` flips on every swap.
static void resolveBuffersInPlace(MutableArrayRef<Value> operands,
                                  bool &changed) {
  for (Value &v : operands) {
    if (!isa<BufferType>(v.getType()))
      continue;
    auto info = resolveKernelArg(v);
    if (!info)
      continue;
    v = info->ptr;
    changed = true;
  }
}

// Swap LDS-backed ins of an `hc.generic` to the underlying
// `!hc.ptr<workgroup, T>`. The launch-body converter maps
// `!hc.bare_tensor` directly to `!hc.ptr<workgroup, T>`, so the
// adaptor hands back the ptr SSA the producer's conversion planted.
// Semantic `!hc.tensor` doesn't appear here by contract —
// `hc-decompose-shaped-values` splits it into bare (data, mask)
// pairs upstream so launch-body only sees the bare carrier.
static void swapInsLDSCarriersToPtrs(MutableArrayRef<Value> newIns,
                                     ValueRange adaptedIns, bool &changed) {
  for (auto [idx, v] : llvm::enumerate(newIns)) {
    if (!isa<BareTensorType>(v.getType()))
      continue;
    Value adapted = adaptedIns[idx];
    if (!isa<PtrType>(adapted.getType()))
      continue;
    newIns[idx] = adapted;
    changed = true;
  }
}

// Resolve a `hc.generic` operand from its post-flatten `!hc.buffer<T,
// ["?"]>` carrier (or a pre-flatten kernel-arg buffer carrier) back
// to the underlying `!hc.ptr<global, T>` produced by the kernel-arg
// UCC chain. Leaves non-buffer operands (`!hc.bare_vector`,
// `!hc.bare_tensor`, workgroup ptrs, ...) untouched — those route
// through other lowering paths.
//
// The pass keeps the iter bounds, the offset arrays, and the body
// verbatim; only the operand SSA values change. The composed offset
// expression already references kernel ABI symbols (e.g.
// `$STRIDE_0_<buf>`); binding those symbols is the responsibility
// of whoever later materializes the offset into SSA — `hc-lower-
// generic` v0 fills that slot via `hc.idx_apply`'s ambient walk.
struct AdaptGenericOp : public OpConversionPattern<HCGenericOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCGenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Two HC-to-HC retypes happen here; the verifier on the other
    // side stays happy because `HC_GenericOperandType` admits both
    // `!hc.ptr` (global ABI ptr, workgroup LDS ptr) and `!hc.undef`.
    // bare_vector deliberately doesn't appear in either retype — the
    // global converter maps it to builtin `vector<NxT>`, which the
    // generic-op verifier rejects, so we keep bare_vector operands at
    // the original SSA and let `hc-lower-generic` consume them with
    // its own value-carrier path.
    //
    //   * Kernel-arg buffers resolve to their `!hc.ptr<global, T>` via
    //     the bundle UCC chain `hc-lower-kernels-to-gpu-launch`
    //     planted. The adaptor doesn't help here because the bundle is
    //     a multi-output UCC — `resolveKernelArg` walks it explicitly.
    //
    //   * bare_tensor ins swap to the `!hc.ptr<workgroup, T>` SSA the
    //     adaptor hands back. This is the boundary `hc.zeros` (and
    //     friends) emit when their bare_tensor result gets lowered to
    //     an LDS allocation: the global converter remembers the
    //     replacement, so `adaptor.getIns()[k]` is the ptr directly
    //     (no UCC to peek through). The legality predicate keys this
    //     off `converter.convertType(operandType)` — a pure type-
    //     level check, no SSA chain inspection.
    //
    // Semantic `!hc.tensor` doesn't appear at this boundary —
    // `hc-decompose-shaped-values` splits it into bare (data, mask)
    // pairs upstream, so launch-body only sees the bare carrier.
    //
    // Outs is left in `bare_tensor` form. Swapping outs
    // would collapse the value-typed SSA result the op contracts to
    // produce (`ptr` outs contribute no SSA result; the verifier
    // matches result count against value-typed outs count). The
    // outs-side LDS swap needs its own rewrite that also rewires the
    // result chain — a separate change.
    //
    // Iter bounds, on the other hand, do accept the index conversion
    // because `HC_ValueType` covers both `!hc.idx<>` and `index`; pull
    // them from the adaptor to clean up the trailing idx-to-index cast
    // chain in one shot.
    SmallVector<Value> newIns(op.getIns());
    SmallVector<Value> newOuts(op.getOuts());
    bool changed = false;
    resolveBuffersInPlace(newIns, changed);
    resolveBuffersInPlace(newOuts, changed);
    swapInsLDSCarriersToPtrs(newIns, adaptor.getIns(), changed);
    if (!changed)
      return failure();

    auto newOp = HCGenericOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), op.getIterSymsAttr(),
        adaptor.getIterBounds(), op.getIterKindsAttr(), ValueRange(newIns),
        ValueRange(newOuts), /*ambient_idxs=*/adaptor.getAmbientIdxs(),
        op.getAmbientIdxSymsAttr(), op.getInsOffsetsAttr(),
        op.getOutsOffsetsAttr());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// Clone an `hc.yield`-terminated region into the (already-created)
// matching `scf` region. Maps block arguments 1:1, clones each
// non-terminator op through `rewriter.clone`, and emits an `scf.yield`
// whose operands are the values from the original `hc.yield`, each
// cast (if needed) to the corresponding `resultTypes` entry.
//
// Used for both `scf.for` (passing the loop's result types) and
// `scf.if` (passing the converted result types) — the cloning pattern
// is identical in both cases.
//
// For `hc.for_range` lowering specifically, the `$joinN` symbolic name
// binding the original induction var carried (`!hc.idx<"$joinN">`) is
// no longer needed past this point: `hc-flatten-with-layouts` captured
// it into every consuming `hc.generic`'s `ambient_idxs` while the
// typed IV was still in scope, so any downstream apply that references
// `$joinN` has the SSA edge operand-bound and survives this conversion
// verbatim.
static LogicalResult cloneHCYieldRegion(Location loc, Region &srcRegion,
                                        Region &dstRegion,
                                        TypeRange resultTypes,
                                        ConversionPatternRewriter &rewriter) {
  Block &src = srcRegion.front();
  Block &dst = dstRegion.front();
  Operation *dstTerminator = dst.empty() ? nullptr : &dst.back();
  if (dstTerminator)
    rewriter.setInsertionPoint(dstTerminator);
  else
    rewriter.setInsertionPointToEnd(&dst);

  IRMapping mapping;
  for (auto [oldArg, newArg] :
       llvm::zip_equal(src.getArguments(), dst.getArguments()))
    mapping.map(oldArg, newArg);

  auto yield = dyn_cast<HCYieldOp>(src.back());
  if (!yield)
    return failure();
  for (Operation &nested : src) {
    if (&nested == yield.getOperation())
      break;
    rewriter.clone(nested, mapping);
  }

  SmallVector<Value> yielded;
  for (auto [index, value] : llvm::enumerate(yield.getValues())) {
    Value mapped = mapping.lookupOrDefault(value);
    yielded.push_back(castIfNeeded(rewriter, loc, mapped, resultTypes[index]));
  }
  if (dstTerminator)
    rewriter.replaceOpWithNewOp<scf::YieldOp>(dstTerminator, yielded);
  else
    scf::YieldOp::create(rewriter, loc, yielded);
  return success();
}

struct ConvertForRangeOp : public OpConversionPattern<HCForRangeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCForRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLower().getType().isIndex() ||
        !adaptor.getUpper().getType().isIndex() ||
        !adaptor.getStep().getType().isIndex())
      return op.emitOpError("requires index lower, upper, and step");

    auto loop = scf::ForOp::create(rewriter, op.getLoc(), adaptor.getLower(),
                                   adaptor.getUpper(), adaptor.getStep(),
                                   adaptor.getIterInits());
    if (failed(cloneHCYieldRegion(op.getLoc(), op.getBody(), loop.getRegion(),
                                  loop.getResultTypes(), rewriter)))
      return op.emitOpError("body must end with `hc.yield`");
    rewriter.replaceOp(op, loop.getResults());
    return success();
  }
};

struct ConvertIfOp : public OpConversionPattern<HCIfOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCIfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> results;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), results)))
      return failure();
    auto ifOp =
        scf::IfOp::create(rewriter, op.getLoc(), results, adaptor.getCond(),
                          /*withElseRegion=*/!op.getElseRegion().empty());
    if (failed(cloneHCYieldRegion(op.getLoc(), op.getThenRegion(),
                                  ifOp.getThenRegion(), results, rewriter)))
      return op.emitOpError("then region must end with `hc.yield`");
    if (!op.getElseRegion().empty() &&
        failed(cloneHCYieldRegion(op.getLoc(), op.getElseRegion(),
                                  ifOp.getElseRegion(), results, rewriter)))
      return op.emitOpError("else region must end with `hc.yield`");
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

static void populateLaunchBodyLoweringPatterns(TypeConverter &converter,
                                               MLIRContext *ctx,
                                               RewritePatternSet &patterns) {
  patterns.add<
      ConvertIdxApplyOp, ConvertPredApplyOp, ConvertConstOp,
      ConvertIntBinaryOp<HCAddOp, arith::AddIOp>,
      ConvertIntBinaryOp<HCSubOp, arith::SubIOp>,
      ConvertIntBinaryOp<HCMulOp, arith::MulIOp>,
      // `hc.and` / `hc.or` on scalar `!hc.pred` arrive here from
      // `hc-load-store-to-generic`, which AND-s the source mask with
      // the dst-bounds predicate inside the generic body. Other
      // shapes flow through `hc-elementwise-to-generic` first and
      // land here as scalar ops on i1 too.
      ConvertIntBinaryOp<HCAndOp, arith::AndIOp>,
      ConvertIntBinaryOp<HCOrOp, arith::OrIOp>,
      ConvertFloatBinaryOp<HCAddOp, arith::AddFOp>,
      ConvertFloatBinaryOp<HCSubOp, arith::SubFOp>,
      ConvertFloatBinaryOp<HCMulOp, arith::MulFOp>, ConvertDivOp,
      ConvertIntBinaryOp<HCModOp, arith::RemUIOp>, ConvertNegOp,
      ConvertCmpOp<HCCmpLtOp>, ConvertCmpOp<HCCmpLeOp>, ConvertCmpOp<HCCmpGtOp>,
      ConvertCmpOp<HCCmpGeOp>, ConvertCmpOp<HCCmpEqOp>, ConvertCmpOp<HCCmpNeOp>,
      ConvertCastOp, ConvertBufferDimOp, ConvertIntrinsicSignatureOp,
      ConvertLoadLikeOp<HCLoadOp>, ConvertLoadLikeOp<HCVLoadOp>,
      ConvertFullMaskOp, ConvertNullaryShapedConstantOp<HCVZerosOp, 0>,
      ConvertNullaryShapedConstantOp<HCVOnesOp, 1>,
      ConvertNullaryShapedConstantOp<HCZerosOp, 0>,
      ConvertNullaryShapedConstantOp<HCOnesOp, 1>,
      ConvertFillShapedConstantOp<HCVFullOp>,
      ConvertFillShapedConstantOp<HCFullOp>, ConvertEmptyOp, ConvertVecOp,
      ConvertSelectOp, ConvertStoreOp, ConvertCallIntrinsicOp,
      ConvertBufferViewOp, ConvertForRangeOp, ConvertIfOp>(converter, ctx);

  patterns.add<AdaptRegionlessOp<HCTupleOp>, AdaptRegionlessOp<HCSliceExprOp>,
               AdaptRegionlessOp<HCGetItemOp>, AdaptGenericOp>(converter, ctx);
}

static bool regionsAreLegal(Operation *op, const TypeConverter &converter) {
  return llvm::all_of(op->getRegions(), [&](Region &region) {
    return converter.isLegal(&region);
  });
}

// An `hc.generic` operand is legal at this pass's boundary unless it
// still references a kernel-arg buffer that hasn't been resolved to a
// `!hc.ptr<global, T>`. `AdaptGenericOp` is responsible for that
// resolution; until it fires, the op stays illegal.
static bool isHCGenericOperandLegal(Value v) {
  return !isa<BufferType>(v.getType()) || !resolveKernelArg(v);
}

static bool isHCGenericLegalAtLaunchBoundary(HCGenericOp op) {
  if (!llvm::all_of(op.getIns(), isHCGenericOperandLegal))
    return false;
  if (!llvm::all_of(op.getOuts(), isHCGenericOperandLegal))
    return false;
  // bare_tensor ins lowers to `!hc.ptr<workgroup, T>` via
  // `AdaptGenericOp`. Type-level check: the launch-body type converter
  // maps bare_tensor unconditionally; the adaptor will hand back the
  // matching ptr SSA the producer's conversion already planted. Outs
  // intentionally not checked here — we don't swap outs (see the
  // pattern comment).
  for (Value v : op.getIns()) {
    if (isa<BareTensorType>(v.getType()))
      return false;
  }
  return true;
}

// `hc.intrinsic` is legal iff its signature already matches the
// converter's idea of its input / result types. A missing signature
// (parse error / orphan IR) is legal so the verifier can flag it on
// its own terms; this predicate only kicks in for well-typed
// intrinsics whose function type is converter-stable.
static bool isHCIntrinsicSignatureLegal(HCIntrinsicOp op,
                                        const TypeConverter &converter) {
  std::optional<FunctionType> fnType = op.getFunctionType();
  if (!fnType)
    return true;
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  if (failed(convertIntrinsicFunctionSignature(*fnType, converter, inputs,
                                               results)))
    return false;
  return llvm::equal(fnType->getInputs(), inputs) &&
         llvm::equal(fnType->getResults(), results);
}

// `hc.call_intrinsic` is legal iff every operand and result type
// already equals the converter's intrinsic-boundary projection. If
// any type can't be projected (boundary returns null) the call is
// illegal — the conversion driver should rewrite it.
static bool isHCCallIntrinsicLegal(HCCallIntrinsicOp op,
                                   const TypeConverter &converter) {
  for (Value arg : op.getArgs()) {
    Type boundary = convertIntrinsicBoundaryType(arg.getType(), converter);
    if (!boundary || arg.getType() != boundary)
      return false;
  }
  for (Type result : op.getResultTypes()) {
    Type boundary = convertIntrinsicBoundaryType(result, converter);
    if (!boundary || result != boundary)
      return false;
  }
  return true;
}

// HC ptr-family ops are produced by this pass (workgroup tiles) and must
// pass through to the downstream `hc-lower-to-llvm` slot. The other
// HC ops in the illegal set are the launch-body surface that lowers
// here; anything from a non-`hc` dialect is unconditionally legal.
static void registerLaunchBodyHCLegality(ConversionTarget &target) {
  // `hc.generic` is dynamically legal: legal once `AdaptGenericOp` has
  // resolved every kernel-arg buffer operand back to its underlying
  // `!hc.ptr<global, T>` (the post-flatten access path expects a ptr
  // operand + composed 1D offset for `hc-lower-generic` v0). Non-buffer
  // operands (`!hc.bare_vector`, `!hc.bare_tensor`, workgroup ptrs)
  // pass through unchanged. `hc.yield_predicated` is the masked-yield
  // terminator for `hc.generic` bodies and rides the same legality
  // slot. `hc.yield` is illegal at the launch-body root (where its
  // historical user is `hc.for_range`, lowered to `scf.for` here) but
  // dynamically legal inside `hc.generic` so the generic-body
  // terminator survives the pass for `hc-lower-generic` to consume.
  target.addLegalOp<HCUndefValueOp, UnrealizedConversionCastOp, HCAllocOp,
                    HCPtrOffsetOp, HCPtrLoadOp, HCPtrStoreOp, HCPtrLoadPredOp,
                    HCPtrStorePredOp, HCYieldPredicatedOp>();
  target.addDynamicallyLegalOp<HCGenericOp>(isHCGenericLegalAtLaunchBoundary);
  target.addDynamicallyLegalOp<HCYieldOp>([](HCYieldOp op) {
    return isa_and_nonnull<HCGenericOp>(op->getParentOp());
  });
  // `hc.load_mask` is illegal here too — `hc-load-store-to-generic`
  // already rewrote every load_mask the front IR can produce into an
  // `hc.generic` body whose predicate is materialised structurally.
  // Anything that survives is a producer bug and the conversion
  // driver fails loudly instead of falling back to the old per-axis
  // post-flatten lowering (which used to clamp every mask to
  // all-false against the placeholder 1D kernel-arg dim).
  target.addIllegalOp<HCConstOp, HCAddOp, HCSubOp, HCMulOp, HCDivOp, HCModOp,
                      HCAndOp, HCOrOp, HCNegOp, HCCmpLtOp, HCCmpLeOp, HCCmpGtOp,
                      HCCmpGeOp, HCCmpEqOp, HCCmpNeOp, HCCastOp, HCBufferDimOp,
                      HCLoadOp, HCVLoadOp, HCLoadMaskOp, HCBufferViewOp,
                      HCVecOp, HCVZerosOp, HCVOnesOp, HCVFullOp, HCFullMaskOp,
                      HCZerosOp, HCOnesOp, HCFullOp, HCEmptyOp, HCSelectOp,
                      HCStoreOp, HCForRangeOp, HCIfOp>();
}

// `hc.idx_apply` / `hc.pred_apply` inside an `hc.generic` body are
// left alone for the second invocation of this pass to consume —
// their iter-sym free names get explicit per-lane bindings only after
// `hc-lower-generic` unrolls the generic. The two-pass dance matches
// `hc.yield`: bodily resident, materialised later. Outside the body
// (kernel-scope offset emission, ambient-binding helpers from earlier
// passes) the ops are illegal and lower here.
static void registerLaunchBodyApplyLegality(ConversionTarget &target) {
  auto applyLegalInsideGeneric = [](Operation *op) {
    return op->getParentOfType<HCGenericOp>() != nullptr;
  };
  target.addDynamicallyLegalOp<HCIdxApplyOp>(applyLegalInsideGeneric);
  target.addDynamicallyLegalOp<HCPredApplyOp>(applyLegalInsideGeneric);
}

static ConversionTarget
makeLaunchBodyLoweringTarget(MLIRContext *ctx, const TypeConverter &converter) {
  ConversionTarget target(*ctx);
  target
      .addLegalDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                       scf::SCFDialect, vector::VectorDialect>();
  registerLaunchBodyHCLegality(target);
  registerLaunchBodyApplyLegality(target);
  target.addDynamicallyLegalOp<HCIntrinsicOp>([&](HCIntrinsicOp op) {
    return isHCIntrinsicSignatureLegal(op, converter);
  });
  target.addDynamicallyLegalOp<HCCallIntrinsicOp>([&](HCCallIntrinsicOp op) {
    return isHCCallIntrinsicLegal(op, converter);
  });
  target.addDynamicallyLegalOp<HCTupleOp, HCSliceExprOp, HCGetItemOp>(
      [&](Operation *op) {
        return converter.isLegal(op) && regionsAreLegal(op, converter);
      });
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) { return !isa<HCDialect>(op->getDialect()); });
  return target;
}

// Contract gate for the launch-body lowering: `hc-decompose-shaped-
// values` runs upstream and splits every semantic `!hc.tensor` /
// `!hc.vector` producer into bare (data, mask) pairs, so by the time
// this pass runs there's no semantic carrier left to lower. If one
// survives — either decompose missed a producer pattern, or someone
// re-introduced the semantic carrier downstream — fail loud with the
// offending op rather than silently identity-converting the type and
// having `applyPartialConversion` produce a vague "failed to legalize"
// error two stages later. Catching it here points the finger at the
// upstream gap, not at this pass.
static bool isSemanticShapedType(Type type) {
  return isa<hc::TensorType, hc::VectorType>(type);
}

static LogicalResult assertNoSemanticShapedSurvives(Operation *rootOp) {
  WalkResult walk = rootOp->walk([&](Operation *op) {
    auto bail = [&](Type type, StringRef role) -> WalkResult {
      op->emitOpError("semantic shaped type ")
          << type << " survived past hc-decompose-shaped-values on " << role
          << "; decompose must split !hc.tensor / !hc.vector into bare "
             "(data, mask) pairs before hc-lower-launch-body runs";
      return WalkResult::interrupt();
    };
    for (Value v : op->getOperands())
      if (isSemanticShapedType(v.getType()))
        return bail(v.getType(), "operand");
    for (Type t : op->getResultTypes())
      if (isSemanticShapedType(t))
        return bail(t, "result");
    return WalkResult::advance();
  });
  return success(!walk.wasInterrupted());
}

// Bare carriers (`!hc.bare_tensor` / `!hc.bare_vector`) collapse to
// `!hc.ptr<workgroup, T>` / `!vector<...>` here, both of which need
// the element count known at compile time — workgroup LDS is a
// fixed per-CU resource and the AMDGPU vector type is sized at the
// MLIR level. A bare carrier whose shape still carries a free symbol
// after `hc-specialize-literals` would identity-convert through the
// type converter (the converter's static-shape predicate fails) and
// then surface as a vague "explicitly marked illegal" rejection on
// the producer (`hc.zeros` / `hc.full` / `hc.load`). Catching it
// here, before the partial conversion, names the offending op and
// the unresolved sym so the user can pin it via
// `hc.compile(symbols={...})` (or by adding it to the kernel's
// `literals=` whitelist) instead of decoding the post-conversion
// failure two stages later. The symbolic-dim ergonomic case is
// tracked separately — eventually a per-thread register-tile path
// would let those bare carriers stay symbolic — but that's a
// distinct architectural change; this gate just keeps the failure
// mode honest in the meantime.
static bool isStaticShape(ShapeAttr shape) {
  if (!shape)
    return true;
  return succeeded(staticIntegerShape(shape, nullptr));
}

static LogicalResult assertBareCarriersAreStaticShape(Operation *rootOp) {
  WalkResult walk = rootOp->walk([&](Operation *op) {
    auto checkType = [&](Type type, StringRef role) -> WalkResult {
      auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
      if (!shaped || !isa<BareTensorType, BareVectorType>(type))
        return WalkResult::advance();
      if (isStaticShape(shaped.getSymbolicShape()))
        return WalkResult::advance();
      op->emitOpError("bare carrier ")
          << type << " has a non-literal shape on " << role
          << "; hc-lower-launch-body needs every dim resolved to an integer "
             "literal to allocate the workgroup tile, so bind the free "
             "symbol(s) via hc.compile(symbols={...}) (add the symbol to the "
             "kernel decorator's `literals=` set if it isn't already)";
      return WalkResult::interrupt();
    };
    for (Value v : op->getOperands())
      if (WalkResult r = checkType(v.getType(), "operand"); r.wasInterrupted())
        return r;
    for (Type t : op->getResultTypes())
      if (WalkResult r = checkType(t, "result"); r.wasInterrupted())
        return r;
    return WalkResult::advance();
  });
  return success(!walk.wasInterrupted());
}

struct HCLowerLaunchBodyPass
    : public hc::impl::HCLowerLaunchBodyBase<HCLowerLaunchBodyPass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(assertNoSemanticShapedSurvives(getOperation())))
      return signalPassFailure();
    if (failed(assertBareCarriersAreStaticShape(getOperation())))
      return signalPassFailure();

    MLIRContext *ctx = &getContext();
    HCLaunchBodyTypeConverter converter;
    RewritePatternSet patterns(ctx);
    populateLaunchBodyLoweringPatterns(converter, ctx, patterns);

    ConversionTarget target = makeLaunchBodyLoweringTarget(ctx, converter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// `createHCLowerLaunchBodyPass()` is emitted by tablegen.
