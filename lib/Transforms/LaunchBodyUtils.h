// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared infrastructure for launch-body lowering: the type converter,
// kernel-arg ABI walker, ambient symbol bindings, and the symbolic
// expression -> arith lowerer. Hoisted out of
// HCLowerLaunchBodyPass.cpp's anonymous namespace so the slice
// passes that grew out of it (`hc-lower-launch-scalar-ops`,
// `hc-lower-launch-shaped-constants`, `hc-lower-launch-memory-access`,
// `hc-lower-apply`, `hc-bridge-intrinsics`, `hc-reconcile-generic-operands`,
// and any future ones) can reach for the same primitives without
// going through the populator indirection.

#ifndef HC_TRANSFORMS_LAUNCH_BODY_UTILS_H
#define HC_TRANSFORMS_LAUNCH_BODY_UTILS_H

#include "hc/IR/HCAttrs.h"
#include "hc/IR/HCSymbols.h"
#include "hc/IR/HCTypes.h"
#include "hc/IR/HCTypesInterfaces.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::hc {

// Symbolic-payload accessors. `rawNode` peels the `const` off an
// `ixs_node *` pointer the attr stores; `exactSymbolName` returns the
// leaf sym name when the payload is a bare `IXS_SYM`.
ixs_node *rawNode(ExprAttr expr);
ixs_node *rawNode(PredAttr pred);
std::optional<StringRef> exactSymbolName(ExprAttr expr);
std::optional<StringRef> exactSymbolName(Type type);

// Element-type projection: `!hc.pred` -> `i1`; otherwise scalar
// arith types pass through; non-arith returns null.
Type convertElementType(Type type);

// Project a `ShapeAttr` to a vector of integer literal dim values.
// Non-static dims fail; `diagOp` non-null enables a per-dim emitOpError.
FailureOr<SmallVector<int64_t>> staticIntegerShape(ShapeAttr shape,
                                                   Operation *diagOp);
FailureOr<SmallVector<int64_t>>
staticIntegerShape(SymbolicallyShapedTypeInterface shaped);

// Static element count of a `!hc.bare_tensor`. Product over static dims.
FailureOr<int64_t> bareTensorElementCount(BareTensorType type);

// `!hc.bare_tensor` -> `!hc.ptr<workgroup, T>` (LDS storage; static
// dims required for `hc.alloc`'s element count).
Type convertBareTensorType(BareTensorType type);

// `!hc.bare_vector<T, [..]>` -> `vector<NxT>`; static dims required.
Type convertBareVectorType(BareVectorType type);

// Ambient symbol scope: launch-geometry / kernel-arg / block-arg
// SSA bindings indexed by symbol name. `BoundValues` is the input
// the `ExprLowerer` resolves leaves against.
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

// Launch-boundary kernel-arg bundle: `(ptr, dim_0..dim_{r-1},
// stride_0..stride_{r-1})` unpacked from the UCC chain
// `hc-lower-kernels-to-gpu-launch` plants. `resolveKernelArg` walks
// back through the chain to fill this in.
struct KernelArgSource {
  Value ptr;
  SmallVector<Value> dims;
  SmallVector<Value> strides;

  unsigned rank() const { return dims.size(); }
};

// Build a fresh UCC bridging `value` to `type` when types differ;
// passthrough when they match.
Value castIfNeeded(OpBuilder &builder, Location loc, Value value, Type type);

// Source / target materialisation helper for the launch-body type
// converter -- emits an `unrealized_conversion_cast` from a single
// input value, with a short-circuit through the kernel-arg bundle
// when the target is `index` and the source is an `!hc.idx<sym>`
// rooted in a bundle UCC.
Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                      Location loc);

// Generic UCC cast to `index`; short-circuits through the bundle
// when the source is `!hc.idx<sym>`.
Value indexCast(OpBuilder &builder, Location loc, Value value);
Value indexCastViaBundle(OpBuilder &builder, Location loc, Value value);

// Walk back through the kernel-arg UCC bundle. Returns std::nullopt
// when `source` isn't a bundle UCC.
std::optional<KernelArgSource> resolveKernelArg(Value source);

// Same as above but specialised for access ops: post-flatten
// single-axis carriers project to a synthetic rank-1 view.
std::optional<KernelArgSource>
resolveAccessKernelArg(OpBuilder &builder, Location loc, Value source);

// Linearise a per-axis index vector into a single flat element
// offset using the kernel-arg strides:
// `sum_axis(indices[a] * strides[a])`.
Value linearizeKernelArgOffset(OpBuilder &builder, Location loc,
                               const KernelArgSource &source,
                               ValueRange indices);

// Collect ambient symbol bindings (launch geometry, kernel-arg UCC,
// ancestor block args) reachable from `anchor`'s surrounding scope.
// The returned `BoundValues` is what `ExprLowerer` consumes.
BoundValues collectBoundValues(Operation *anchor,
                               ConversionPatternRewriter &rewriter);

// Augment ambient bindings with per-operand bindings produced by an
// apply op's `(symbols, operands)` zip. Operand-side overrides
// ambient on collision.
BoundValues collectApplyBindings(Operation *op,
                                 ConversionPatternRewriter &rewriter,
                                 ArrayAttr symbols,
                                 ValueRange convertedOperands);

// Lower a symbolic `#hc.expr` / `#hc.pred` payload to `index` /
// `i1` SSA via `arith.*` / `arith.cmpi` / `arith.and|or` chains.
// Leaves get resolved against `BoundValues`; first unresolved leaf
// is captured for diagnostics (`lastUnresolvedSymbol`).
class ExprLowerer {
public:
  ExprLowerer(OpBuilder &builder, Location loc, const BoundValues &boundValues);

  FailureOr<Value> lower(ExprAttr expr);
  FailureOr<Value> lower(PredAttr pred);

  Value constant(int64_t value);

  // First leaf the lowerer couldn't bind, or empty if all bound. See
  // `doc/layouts.md` "Free symbols in layout offsets".
  StringRef lastUnresolvedSymbol() const { return unresolvedSymbol; }

private:
  FailureOr<Value> lowerNode(ixs_node *node);
  FailureOr<Value> lowerPredNode(ixs_node *node);
  FailureOr<Value> lowerRational(ixs_node *node);
  FailureOr<Value> lowerSymbol(ixs_node *node);
  FailureOr<Value> lowerAdd(ixs_node *node);
  FailureOr<Value> scaleTerm(ixs_node *coeff, Value term);
  FailureOr<Value> lowerMul(ixs_node *node);
  FailureOr<Value> lowerCeil(ixs_node *node);
  FailureOr<std::pair<Value, Value>> lowerAsFraction(ixs_node *node);
  FailureOr<Value> lowerCmp(ixs_node *node);

  template <typename OpT> FailureOr<Value> lowerBinary(ixs_node *node);
  template <typename OpT> FailureOr<Value> lowerLogic(ixs_node *node);

  OpBuilder &builder;
  Location loc;
  const BoundValues &boundValues;
  std::string unresolvedSymbol;
};

// Type converter shared by every launch-body lowering pass:
// `!hc.idx` -> `index`, `!hc.pred` -> `i1`, bare carriers ->
// `vector<NxT>` / `!hc.ptr<workgroup, T>`, tuples recurse,
// source/target materialisations via `materializeCast`.
class HCLaunchBodyTypeConverter : public TypeConverter {
public:
  HCLaunchBodyTypeConverter();
};

} // namespace mlir::hc

#endif // HC_TRANSFORMS_LAUNCH_BODY_UTILS_H
