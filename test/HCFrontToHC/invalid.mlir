// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative-path coverage: every top-level module below expects a
// specific diagnostic so the pass's guardrails stay wired. `--split-input-file`
// keeps each module independent; `-verify-diagnostics` matches the
// `expected-error` annotations against the actual diagnostics.
// RUN: hc-opt --convert-hc-front-to-hc --split-input-file -verify-diagnostics %s

// Missing `parameters` attribute: the driver is required to stamp one,
// so a bare kernel op without it surfaces a hard error.
module {
  // expected-error@+1 {{missing `parameters` attribute}}
  hc_front.kernel "bad_no_params" {
    hc_front.return
  }
}

// -----

// Note: a "param" name that never gets bound is no longer a
// conversion-time error. This pass emits `hc.name_load "x"`
// unconditionally; a name with no reaching `hc.assign` surfaces in
// `-hc-promote-names`, which owns the diagnostic path. The exact
// wording is pinned in `test/HC/promote-names-invalid.mlir`
// (promote-names in isolation) and `test/HCFrontToHC/pipeline-invalid.mlir`
// (the end-to-end conversion + promote pipeline, i.e. the supported
// user-facing shape). Keeping the coverage split that way lets each
// pass assert its own contract.

// -----

// Malformed hand-written IR can expose a block argument that the converter
// never mapped from an hc_front producer. Diagnose the missing lowering at the
// consumer instead of tripping the internal lowering map assertion.
module {
  hc_front.kernel "unmapped_block_arg_return" attributes {parameters = []} {
  ^bb0(%arg0: !hc_front.value):
    // expected-error@+1 {{return operand was not lowered before use}}
    hc_front.return %arg0
  }
}

// -----

// Unknown binop kind — `Pow` and friends are unsupported today.
module {
  hc_front.kernel "bad_binop" attributes {
    parameters = [{name = "a"}, {name = "b"}]
  } {
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    // expected-error@+1 {{unsupported hc_front.binop kind 'Pow'}}
    %c = hc_front.binop "Pow"(%a, %b)
    hc_front.return
  }
}

// -----

// Out-of-range launch-geometry axis. `kMaxLaunchAxis` (32) is the cap;
// anything at or above is rejected before the unsigned cast.
module {
  hc_front.kernel "bad_axis" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %gid = hc_front.attr %grp, "group_id" {ref = {kind = "dsl_method", method = "group_id"}}
    %ax = hc_front.constant<99 : i64>
    // expected-error@+1 {{launch-geo axis 99 out of range}}
    %out = hc_front.subscript %gid[%ax]
    %t = hc_front.target_name "t"
    hc_front.assign %t = %out
    hc_front.return
  }
}

// -----

// Scalar launch-geometry queries are values, not per-axis tuples.
module {
  hc_front.kernel "bad_scalar_axis" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %group_size = hc_front.attr %grp, "group_size" {ref = {kind = "dsl_method", method = "group_size"}}
    %ax = hc_front.constant<0 : i64>
    // expected-error@+1 {{scalar launch-geo query 'group_size' is not subscriptable}}
    %out = hc_front.subscript %group_size[%ax]
    %t = hc_front.target_name "t"
    hc_front.assign %t = %out
    hc_front.return
  }
}

// -----

// The call-style spelling must reject scalar subscripts too.
module {
  hc_front.kernel "bad_scalar_call_axis" attributes {
    parameters = [{name = "group"}]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %wave_size = hc_front.attr %grp, "wave_size" {ref = {kind = "dsl_method", method = "wave_size"}}
    %wave = hc_front.call %wave_size()
    %ax = hc_front.constant<0 : i64>
    // expected-error@+1 {{scalar launch-geo query 'wave_size' is not subscriptable}}
    %out = hc_front.subscript %wave[%ax]
    %t = hc_front.target_name "t"
    hc_front.assign %t = %out
    hc_front.return
  }
}

// -----

// Unsupported DSL method — spelled correctly enough to reach dsl dispatch
// but not one we implement.
module {
  hc_front.kernel "bad_dsl" attributes {parameters = [{name = "x"}]} {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %m = hc_front.attr %x, "no_such_method" {ref = {kind = "dsl_method", method = "no_such_method"}}
    // expected-error@+1 {{unsupported dsl_method 'no_such_method'}}
    %v = hc_front.call %m()
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
    hc_front.return
  }
}

// -----

// Slice operand count disagrees with has_* flags — hand-crafted inconsistency.
module {
  hc_front.kernel "bad_slice_arity" attributes {parameters = [{name = "a"}]} {
    %c0 = hc_front.constant<0 : i64>
    %c1 = hc_front.constant<1 : i64>
    %c2 = hc_front.constant<2 : i64>
    // expected-error@+1 {{slice operand count 3 does not match}}
    %s = hc_front.slice (%c0, %c1, %c2) {has_lower = true, has_upper = true, has_step = false}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %out = hc_front.subscript %a[%s]
    %t = hc_front.target_name "t"
    hc_front.assign %t = %out
    hc_front.return
  }
}

// -----

// Unary DSL method with a base that didn't lower (the attr base is a
// callee-classified name, so `lowerValueOperand` returns null). `requireBase`
// must catch this and diagnose rather than build a null-operand hc op.
module {
  hc_front.kernel "bad_null_base_dsl" attributes {parameters = []} {
    // A name with `kind = "builtin"` lowers to null (consumed by parent).
    // Feeding it to `x.vec()` exercises the requireBase guard.
    %n = hc_front.name "some_builtin" {ctx = "load", ref = {kind = "builtin", builtin = "some_builtin"}}
    %m = hc_front.attr %n, "vec" {ref = {kind = "dsl_method", method = "vec"}}
    // expected-error@+1 {{vec: base did not lower}}
    %v = hc_front.call %m()
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
    hc_front.return
  }
}

// -----

// Malformed `ref` on a name: present dict but no (or non-string) `kind`.
// The diagnostic fingers the driver rather than silently returning a
// sentinel null.
module {
  hc_front.kernel "bad_ref_name" attributes {parameters = []} {
    // expected-error@+1 {{`ref` dict with missing or non-string `kind`}}
    %n = hc_front.name "x" {ctx = "load", ref = {notkind = "oops"}}
    hc_front.return
  }
}

// -----

// Same malformed-ref guardrail on an attr op: a dict without a string
// `kind` must blame the attr at source, not the downstream call or
// subscript trying to read `method`.
module {
  hc_front.kernel "bad_ref_attr" attributes {parameters = [{name = "x"}]} {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    // expected-error@+1 {{`ref` dict with missing or non-string `kind`}}
    %m = hc_front.attr %x, "foo" {ref = {method = "foo"}}
    hc_front.return
  }
}

// -----

// `ref.kind = "inline"` is the exclusive business of `-hc-front-inline`;
// if one survives to the conversion boundary the pass ordering is wrong
// and we want a located error rather than a silent placeholder.
module {
  hc_front.kernel "inline_not_inlined" attributes {parameters = [{name = "x"}]} {
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    %inl = hc_front.name "helper" {ctx = "load", ref = {kind = "inline", qualified_name = "pkg.helper"}}
    // expected-error@+1 {{`ref.kind = "inline"` call survived to conversion; run `-hc-front-inline` before `-convert-hc-front-to-hc`}}
    %v = hc_front.call %inl(%x)
    %t = hc_front.target_name "t"
    hc_front.assign %t = %v
    hc_front.return
  }
}

// -----

// A call whose callee is a `ref.kind = "local"` name is the ghost trail
// Python emits for `@group.workitems def inner(...); inner()`. The
// `-hc-front-fold-region-defs` pass erases it upstream; if one
// survives to the converter the pipeline ordering is wrong — the
// diagnostic points directly at the missing pass, parallel to the
// inline case above.
module {
  hc_front.kernel "local_callee_survives" attributes {parameters = []} {
    hc_front.workitem_region attributes {decorators = ["group.workitems"], name = "inner", parameters = [{name = "wi"}]} {
    }
    %0 = hc_front.name "inner" {ctx = "load", ref = {kind = "local"}}
    // expected-error@+1 {{`ref.kind = "local"` call survived to conversion; run `-hc-front-fold-region-defs` before `-convert-hc-front-to-hc`}}
    %1 = hc_front.call %0()
    hc_front.return
  }
}

// -----

// `numpy_dtype_type` with a dtype name the converter doesn't know about
// (`float128` isn't in `resolveNumpyDtypeType` because it's platform-
// dependent and not part of HC's supported scalar set). The attr lowering
// rejects it here rather than later at the call site: the diagnostic
// should finger the attr op, which is where the offending string lives.
module {
  hc_front.kernel "bad_numpy_dtype" attributes {parameters = []} {
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    // expected-error@+1 {{unsupported numpy_dtype_type 'float128'}}
    %bad = hc_front.attr %np, "float128" {ref = {dtype = "float128", kind = "numpy_dtype_type"}}
    hc_front.return
  }
}

// -----

// `np.<dtype>(x)` with an SSA value is not a literal constructor. The
// frontend should spell this as `x.astype(np.<dtype>)`; otherwise returning
// the dtype handle hides the source mistake until a later verifier error.
module {
  hc_front.kernel "numpy_dtype_nonliteral" attributes {
    parameters = [{name = "x"}]
  } {
    %np = hc_front.name "np" {ctx = "load", ref = {kind = "module", module = "numpy"}}
    %f16 = hc_front.attr %np, "float16" {ref = {dtype = "float16", kind = "numpy_dtype_type"}}
    %x = hc_front.name "x" {ctx = "load", ref = {kind = "param"}}
    // expected-error@+1 {{numpy dtype constructor `np.float16(...)` only accepts literal positional arguments in hc_front; use `value.astype(np.float16)` for SSA values}}
    %bad = hc_front.call %f16(%x)
    hc_front.return
  }
}

// -----

// Buffer parameter metadata is part of the conversion boundary now: malformed
// records should fail here, before later passes see partially typed IR.
module {
  // expected-error@+1 {{buffer parameter 'x' is missing `shape` metadata}}
  hc_front.kernel "buffer_missing_shape" attributes {
    parameters = [
      {kind = "buffer", name = "x"}
    ]
  } {
    hc_front.return
  }
}

// -----

// `dtype` is optional, but if present it must name a supported NumPy dtype.
module {
  // expected-error@+1 {{buffer parameter 'x' has unsupported dtype 'float128'}}
  hc_front.kernel "buffer_bad_dtype" attributes {
    parameters = [
      {dtype = "float128", kind = "buffer", name = "x", shape = ["M"]}
    ]
  } {
    hc_front.return
  }
}

// -----

// Shape metadata only has meaning for buffers. Other parameter kinds keep
// using `!hc.undef` until inference can refine them.
module {
  // expected-error@+1 {{parameter 'x' has `shape` metadata but kind 'scalar' is not `buffer`}}
  hc_front.kernel "shape_without_buffer_kind" attributes {
    parameters = [
      {kind = "scalar", name = "x", shape = ["M"]}
    ]
  } {
    hc_front.return
  }
}

// -----

// `group.load(a[i][j], ...)` — chained Python subscripts lower to
// nested `hc.buffer_view`s whose index lists cannot be safely spliced
// (outer slice re-indexes the already-reduced view, not the original
// buffer). `peelBufferView` handles the single-level case; when it
// leaves a `buffer_view` behind, `lowerDslMethodCall` surfaces the
// rewrite suggestion before we emit wrong IR. Store shares the path;
// `vload` uses the same guard but covering one method variant is
// enough to pin the diagnostic wording.
module {
  hc_front.kernel "chained_subscript_load" attributes {
    parameters = [
      {name = "group"},
      {annotation = "Buffer[M,K]", kind = "buffer", name = "a", shape = ["M", "K"]}
    ]
  } {
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %load_attr = hc_front.attr %grp, "load" {ref = {kind = "dsl_method", method = "load"}}
    %zero = hc_front.constant<0 : i64>
    %m = hc_front.constant<16 : i64>
    %k = hc_front.constant<16 : i64>
    %row_sl = hc_front.slice (%zero, %m) {has_lower = true, has_upper = true, has_step = false}
    %col_sl = hc_front.slice (%zero, %k) {has_lower = true, has_upper = true, has_step = false}
    %sub1 = hc_front.subscript %a[%row_sl]
    %sub2 = hc_front.subscript %sub1[%col_sl]
    %m_dim = hc_front.constant<16 : i64>
    %k_dim = hc_front.constant<16 : i64>
    %shape_tuple = hc_front.tuple(%m_dim, %k_dim)
    %shape_kw = hc_front.keyword "shape" = %shape_tuple
    // expected-error@+1 {{chained subscript into `load` is not supported}}
    %tile = hc_front.call %load_attr(%sub2, %shape_kw)
    %t_tile = hc_front.target_name "a_tile"
    hc_front.assign %t_tile = %tile
    hc_front.return
  }
}

// -----

// Intrinsic call missing a declared non-const kwarg. The lowering
// reads the callee's declared `parameters` and diagnoses at the call
// site rather than dropping the kwarg and producing an under-arity
// `hc.call_intrinsic`.
module {
  hc_front.intrinsic "needy" attributes {
    const_kwargs = ["arch"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "group", passing = "positional"},
      {name = "lane", passing = "keyword_only"},
      {name = "arch", passing = "keyword_only"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  hc_front.func "missing_nonconst_kwarg" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "needy" {ctx = "load", ref = {
      callee = "@needy",
      const_kwargs = ["arch"],
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %arch = hc_front.constant<"gfx11">
    %kw_arch = hc_front.keyword "arch" = %arch
    // expected-error@+1 {{intrinsic '@needy' missing required kwarg 'lane'}}
    %r = hc_front.call %intr(%grp, %kw_arch)
    hc_front.return %r
  }
}

// -----

// Intrinsic call with a kwarg name that isn't in the callee's declared
// parameter list (and isn't a const_kwarg either). Unknown names
// surface as a diagnostic at lowering time.
module {
  hc_front.intrinsic "strict" attributes {
    const_kwargs = ["arch"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "group", passing = "positional"},
      {name = "lane", passing = "keyword_only"},
      {name = "arch", passing = "keyword_only"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  hc_front.func "unknown_kwarg" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "lane"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "strict" {ctx = "load", ref = {
      callee = "@strict",
      const_kwargs = ["arch"],
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %lane = hc_front.name "lane" {ctx = "load", ref = {kind = "param"}}
    %kw_lane = hc_front.keyword "lane" = %lane
    %arch = hc_front.constant<"gfx11">
    %kw_arch = hc_front.keyword "arch" = %arch
    %zero = hc_front.constant<0 : i64>
    // `typo` is neither a declared parameter nor a const_kwarg.
    %kw_typo = hc_front.keyword "typo" = %zero
    // expected-error@+1 {{intrinsic '@strict' called with unknown keyword argument 'typo'}}
    %r = hc_front.call %intr(%grp, %kw_lane, %kw_arch, %kw_typo)
    hc_front.return %r
  }
}

// -----

// Frontend intrinsic declarations must carry the Python argument-passing
// class for every parameter so conversion can enforce keyword-only slots.
module {
  // expected-error@+1 {{`parameters` entry at index 0 missing `passing` key}}
  hc_front.intrinsic "unstamped_passing" attributes {
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [{name = "group"}],
    scope = "WorkItem"
  } {
    hc_front.return
  }
}

// -----

// The frontend emits all positional parameters before keyword-only
// parameters. Hand-written IR with a positional parameter after a
// keyword-only one cannot be projected onto the lowered ABI.
module {
  // expected-error@+1 {{positional parameter 'later' cannot follow a keyword-only parameter}}
  hc_front.intrinsic "bad_parameter_order" attributes {
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "first_kw", passing = "keyword_only"},
      {name = "later", passing = "positional"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }
}

// -----

// A keyword-only intrinsic parameter must not be consumed by the positional
// prefix just because it appears in the declared order.
module {
  hc_front.intrinsic "kw_only_lane" attributes {
    const_kwargs = ["arch"],
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "group", passing = "positional"},
      {name = "lane", passing = "keyword_only"},
      {name = "arch", passing = "keyword_only"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  hc_front.func "keyword_only_passed_positionally" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}, {name = "lane"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "kw_only_lane" {ctx = "load", ref = {
      callee = "@kw_only_lane",
      const_kwargs = ["arch"],
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %lane = hc_front.name "lane" {ctx = "load", ref = {kind = "param"}}
    %arch = hc_front.constant<"gfx11">
    %kw_arch = hc_front.keyword "arch" = %arch
    // expected-error@+1 {{intrinsic '@kw_only_lane' parameter 'lane' is keyword-only and cannot be passed positionally}}
    %r = hc_front.call %intr(%grp, %lane, %kw_arch)
    hc_front.return %r
  }
}

// -----

// Intrinsic keyword operands are reserved for parameters declared after
// Python's `*` marker; positional-or-keyword parameters must be passed in the
// positional prefix so the lowered operand ABI stays source-order independent.
module {
  hc_front.intrinsic "two_positionals" attributes {
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "a", passing = "positional"},
      {name = "b", passing = "positional"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  hc_front.func "positional_passed_as_keyword" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "a"}, {name = "b"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "two_positionals" {ctx = "load", ref = {
      callee = "@two_positionals",
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    %kw_b = hc_front.keyword "b" = %b
    // expected-error@+1 {{intrinsic '@two_positionals' parameter 'b' is positional and cannot be passed as a keyword}}
    %r = hc_front.call %intr(%a, %kw_b)
    hc_front.return %r
  }
}

// -----

// More positional operands than the callee has parameters. Caught at
// lowering because the frontend knows the declared shape and can
// point at the offending call site directly.
module {
  hc_front.intrinsic "two_params" attributes {
    decorators = ["kernel.intrinsic"],
    effects = "pure",
    parameters = [
      {name = "a", passing = "positional"},
      {name = "b", passing = "positional"}
    ],
    scope = "WorkItem"
  } {
    hc_front.return
  }

  hc_front.func "too_many_positional" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "a"}, {name = "b"}, {name = "c"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "two_params" {ctx = "load", ref = {
      callee = "@two_params",
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %a = hc_front.name "a" {ctx = "load", ref = {kind = "param"}}
    %b = hc_front.name "b" {ctx = "load", ref = {kind = "param"}}
    %c = hc_front.name "c" {ctx = "load", ref = {kind = "param"}}
    // expected-error@+1 {{intrinsic '@two_params' declares 2 parameter(s), call site supplies 3 positional}}
    %r = hc_front.call %intr(%a, %b, %c)
    hc_front.return %r
  }
}

// -----

// Intrinsic call whose callee is not defined in this module. Conversion
// refuses immediately instead of reconstructing operand/const-kwarg
// metadata from the frontend ref dict.
module {
  hc_front.func "no_stamped_parameters" attributes {
    decorators = ["kernel.func"],
    parameters = [{name = "group"}],
    scope = "WorkItem"
  } {
    %intr = hc_front.name "unstamped" {ctx = "load", ref = {
      callee = "@unstamped",
      effects = "pure",
      kind = "intrinsic",
      scope = "WorkItem"
    }}
    %grp = hc_front.name "group" {ctx = "load", ref = {kind = "param"}}
    %zero = hc_front.constant<0 : i64>
    %kw = hc_front.keyword "extra" = %zero
    // expected-error@+1 {{intrinsic '@unstamped' does not resolve to a lowered hc.intrinsic}}
    %r = hc_front.call %intr(%grp, %kw)
    hc_front.return %r
  }
}
