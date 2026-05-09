// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HC_TRANSFORMS_PASSES
#define HC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HCPromoteNames : Pass<"hc-promote-names"> {
  let summary = "Promote Python name-store ops to SSA values.";
  let description = [{
    Replaces `hc.assign` / `hc.name_load` with structured SSA carriers,
    rebuilding region-carrying ops so each Python-level name becomes a
    real SSA value or iter_arg / result / block-arg of the enclosing
    region op. This pass is the second half of the pipeline; the
    pipeline contract — what the placeholders are, who is allowed to
    look at them, what "supported invocation" means — lives on
    `ConvertHCFrontToHC` in `include/hc/Conversion/HCFrontToHC/Passes.td`.

    Runs bottom-up on every function region tree (`hc.kernel` / `hc.func`
    / `hc.intrinsic`). Region ops that tag themselves with
    `NameStoreRegionOpInterface` participate:

      - `hc.for_range`  : carried names become iter_args + yielded
                          values. A leading `hc.assign "<n>", %iv`
                          (the frontend's way of publishing the IV
                          under its Python-source name) is recognized
                          as a tautological self-bind, kept loop-local,
                          and folded to direct uses of the block arg.
      - `hc.if`         : carried names become op results + two-branch
                          yields; branches must agree on the write set.
      - `hc.workitem_region` / `hc.subgroup_region` : Python-style
                          nested scopes. Reads inside capture outer
                          bindings (lazy outer snapshot); writes
                          shadow without leaking. Out-bound
                          propagation is opt-in via an explicit
                          `hc.region_return <names>` terminator: the
                          promoter rebuilds the op with one result
                          per returned name and pushes writebacks out
                          to the enclosing scope's name store.

    Post-condition (on success): no `hc.assign` / `hc.name_load` survives
    anywhere under the processed callables. Cross-kernel / cross-function
    name capture is out of scope; each callable is its own promotion
    root.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInferTypes : Pass<"hc-infer-types"> {
  let summary = "Infer concrete HC value types from pre-inference `!hc.undef` IR.";
  let description = [{
    Runs a sparse forward dataflow analysis over `hc` IR and rewrites values
    whose type can be refined without changing program structure. The first
    slice focuses on scalar/index facts: constants, symbols, launch geometry,
    slice construction, buffer dimensions, index arithmetic/comparisons, and
    structured loop carriers. Inference is monotonic: `!hc.undef` refines to a
    concrete type, and conflicting concrete facts fail the pass instead of
    being reconciled with a guess.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCVerifyStaticShapes : Pass<"hc-verify-static-shapes"> {
  let summary = "Verify SSA shape operands have static HC shape metadata.";
  let description = [{
    Checks `hc.load`, `hc.vload`, and shaped allocator ops after type
    inference. Shape operands must be tuple values whose element types are
    pinned `!hc.idx<...>` dimensions, and the resulting tensor/vector type must
    carry the same static shape. Load-like ops also require a statically-ranked
    source and index/slice operands whose arity stays within that source rank.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCDecomposeShapedValues
    : Pass<"hc-decompose-shaped-values"> {
  let summary = "Split semantic tensor/vector values into bare data and masks.";
  let description = [{
    Rewrites supported semantic `!hc.tensor` / `!hc.vector` producers and users
    into compiler-internal `!hc.bare_tensor` / `!hc.bare_vector` data plus
    matching bare `!hc.pred` masks. This is an HC-to-HC normalization boundary
    for upstream lowering; semantic shaped values that cross unsupported ops are
    diagnosed instead of being silently dropped.
  }];
  let options = [
    Option<"strictMode", "strict", "bool", "true",
           "Fail unless every semantic shaped boundary is decomposed. Disable "
           "this to run a partial conversion that preserves unsupported "
           "boundaries with unrealized conversion casts.">
  ];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerBarePredicates
    : Pass<"hc-lower-bare-predicates"> {
  let summary = "Lower bare HC predicate containers to upstream i1 masks.";
  let description = [{
    Converts concrete `!hc.pred` values to `i1` and static `!hc.bare_vector`
    values to upstream `vector` values. Predicate vectors become
    `vector<...xi1>` masks, and supported HC mask operations are lowered to
    `arith`/`vector` operations. Bare tensors remain HC values for the later
    memory/tensor lowering. Unsupported HC boundaries are preserved with
    `builtin.unrealized_conversion_cast` so this pass can be staged before the
    full HC-to-upstream lowering.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::hc::HCDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def HCMaterializeBoundExprs
    : Pass<"hc-materialize-bound-exprs"> {
  let summary = "Materialize bound symbolic HC expressions as SSA values.";
  let description = [{
    Replaces uses of values typed as pinned `!hc.idx<expr>` or
    `!hc.pred<pred>` with `hc.materialize_bound_expr` when every symbolic leaf
    is a bound symbol or the expression is constant-only. The original producer
    is left in place for canonicalization/DCE, which lets this pass break
    dependence on launch-context query chains before scope normalization.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCInlineHelpers
    : Pass<"hc-inline-helpers"> {
  let summary = "Inline supported HC helper calls.";
  let description = [{
    Inlines `hc.call` sites targeting `hc.func` helpers. The pass owns HC
    helper call-graph concerns separately from scope-region normalization:
    recursion and unsupported callee body shapes are diagnosed here, and helpers
    that were actually inlined are erased once no `hc.call` references remain.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCNormalizeScopeRegions
    : Pass<"hc-normalize-scope-regions"> {
  let summary = "Flatten supported collective scope regions into SPMD HC IR.";
  let description = [{
    Normalizes HC toward per-workitem SPMD form before upstream GPU lowering.
    The first supported slice flattens resultless `hc.workitem_region` bodies
    after bound expression materialization has removed live uses of the
    workitem token. Result-producing workitem regions and subgroup regions are
    diagnosed until distributed value projection and subgroup-local workitem
    semantics are handled explicitly.
  }];
  let dependentDialects = ["::mlir::hc::HCDialect"];
}

def HCLowerKernelsToGPULaunch
    : Pass<"hc-lower-kernels-to-gpu-launch"> {
  let summary = "Wrap HC kernels in upstream GPU launch operations.";
  let description = [{
    Rewrites `hc.kernel` symbols into host `func.func` entry points that launch
    the former kernel body with `gpu.launch`. Kernel buffer ABI arguments are
    exposed as dynamic memrefs, scalar bound arguments become upstream scalar
    values, and the original HC body is kept inside the launch behind explicit
    cast boundaries for the later memory and intrinsic lowering passes.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::hc::HCDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def HCInterpretIntrinsicRecipes
    : Pass<"hc-interpret-intrinsic-recipes", "::mlir::ModuleOp"> {
  let summary = "Apply embedded intrinsic transform recipes to payload IR.";
  let description = [{
    Looks up a sibling top-level `builtin.module @__hc_intrinsic_lowerings__`
    (carrying `transform.with_named_sequence`) and applies every
    `transform.named_sequence` whose `hc.target` matches the requested target
    string against the surrounding module. The recipes were planted by the
    frontend emitter (`hc/_frontend_mlir.py`) and stick to MLIR transform IR
    end-to-end — no string round-trip.

    Recipes are expected to follow the standard hc transform-op shape: a
    `transform.hc.match_intrinsic_call` to find the call site, optional
    `transform.hc.get_intrinsic_*` handle ops to thread operands, result
    types, and attributes into a `transform.hc.create_op`, then a
    `transform.hc.replace_intrinsic_call` that swaps the original
    `hc.call_intrinsic` for the freshly constructed payload.

    After applying every selected sequence, the lowerings module is erased
    and the pass walks the payload one final time. Any surviving
    `hc.call_intrinsic` is a hard error: the user asked for target `T` but
    nothing under `__hc_intrinsic_lowerings__` knew how to lower the call,
    so we surface the gap here instead of letting it slip into the next
    pass as an unrecognized op.
  }];
  let options = [
    Option<"target", "target", "std::string", "\"\"",
           "Filter named sequences by their `hc.target` attribute. The "
           "empty string runs every sequence and is mostly useful for "
           "single-target tests.">
  ];
  let dependentDialects = [
    "::mlir::hc::HCDialect",
    "::mlir::transform::TransformDialect"
  ];
}

def HCLowerGPUToBinary
    : Pass<"hc-lower-gpu-to-binary", "::mlir::ModuleOp"> {
  let summary = "Compile each gpu.module to a HSACO blob via our pinned ld.lld.";
  let description = [{
    For every `gpu.module` carrying exactly one `#rocdl.target<...>`
    attribute the pass:

      1. translates the module to LLVM IR (`translateModuleToLLVMIR`),
      2. builds an `llvm::TargetMachine` from the rocdl target's triple/
         chip/features and pins the IR's data layout + triple,
      3. runs the standard optimize-then-codegen pipeline at the target's
         opt level (`makeOptimizingTransformer`),
      4. translates LLVM IR to ISA text
         (`LLVM::ModuleToObject::translateModuleToISA`),
      5. assembles ISA → ELF object via `ROCDL::assembleIsa`,
      6. links the ELF → HSACO via `ROCDL::linkObjectCode`, calling our
         pinned `ld.lld`.

    The result is attached as a `gpu::ObjectAttr` on a sibling
    `gpu.binary` op (CompilationTarget::Binary, with the original
    rocdl target preserved), and the source `gpu.module` is erased.

    `lld-path` resolves in this order:

      1. the option's value if non-empty,
      2. the `HC_LLD` environment variable (escape hatch for direct
         `hc-opt` invocations against a non-default toolchain),
      3. the first `ld.lld` on `$PATH`.

    The Python pipeline driver (`hc.compile`) always populates option
    1 from the bundled `hc/_native/bin/ld.lld`, so `HC_LLD` and `$PATH`
    only matter for callers reaching the pass directly via `hc-opt`.
    No ROCm install fallback — if none of the three succeed we error
    out rather than silently picking up a host's distro-installed lld.

    v0 deliberately does not link rocm-device-libs bitcode. The WMMA
    example doesn't reference `__ocml_*` / `__ockl_*`; kernels that do
    will need a follow-up pass that resolves the device-libs path and
    feeds bitcode files into the optimization pipeline before codegen.
  }];
  let options = [
    Option<"lldPath", "lld-path", "std::string", "\"\"",
           "Explicit path to ld.lld; falls back to HC_LLD then $PATH "
           "if empty.">
  ];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::ROCDL::ROCDLDialect"
  ];
}

def HCLowerLaunchBody
    : Pass<"hc-lower-launch-body"> {
  let summary = "Lower HC launch-body operations to upstream dialects.";
  let description = [{
    Converts the HC subset that remains inside `gpu.launch` after kernel
    wrapping. Bound symbolic expressions become upstream SSA values derived
    from launch ids and buffer dimensions, simple arithmetic lowers to `arith`,
    `hc.for_range`/`hc.if` become `scf` control flow, static bare tensors lower
    to workgroup `memref` storage, and bare vectors lower to upstream `vector`
    values. Final masked stores lower to guarded memref writes. Intrinsic
    boundaries remain HC operations with explicit casts until their dedicated
    lowering slice.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::hc::HCDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // HC_TRANSFORMS_PASSES
