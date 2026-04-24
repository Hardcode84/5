// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-c/Front.h"
#include "hc-c/HC.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace hc_front {

struct PyValueType : PyConcreteType<PyValueType> {
  static constexpr IsAFunctionTy isaFunction = mlirHCTypeIsAFrontValueType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirHCFrontValueTypeGetTypeID;
  static constexpr const char *pyClassName = "ValueType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyValueType(context->getRef(),
                             mlirHCFrontValueTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct PyTypeExprType : PyConcreteType<PyTypeExprType> {
  static constexpr IsAFunctionTy isaFunction = mlirHCTypeIsAFrontTypeExprType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirHCFrontTypeExprTypeGetTypeID;
  static constexpr const char *pyClassName = "TypeExprType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyTypeExprType(context->getRef(), mlirHCFrontTypeExprTypeGet(
                                                       context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

} // namespace hc_front
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_hcFrontDialectsNanobind, m) {
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;
  auto hcFrontM = m.def_submodule("hc_front");

  hcFrontM.def(
      "register_dialects",
      [](DefaultingPyMlirContext context, bool load) {
        MlirDialectHandle hcFrontDialect = mlirGetDialectHandle__hc_front__();
        MlirContext context_ = context.get()->get();
        mlirDialectHandleRegisterDialect(hcFrontDialect, context_);
        if (load) {
          mlirDialectHandleLoadDialect(hcFrontDialect, context_);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  using namespace hc_front;
  PyValueType::bind(hcFrontM);
  PyTypeExprType::bind(hcFrontM);

  // `hc` submodule mirrors `hc_front`'s shape but only needs dialect
  // registration today — the hc dialect has no Python-side type constructors
  // we want to expose yet. The name matches the Python shim in
  // `hc_mlir/dialects/hc.py`, which re-exports everything we bind here.
  auto hcM = m.def_submodule("hc");

  hcM.def(
      "register_dialects",
      [](DefaultingPyMlirContext context, bool load) {
        MlirDialectHandle hcDialect = mlirGetDialectHandle__hc__();
        MlirContext context_ = context.get()->get();
        mlirDialectHandleRegisterDialect(hcDialect, context_);
        if (load) {
          mlirDialectHandleLoadDialect(hcDialect, context_);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  // Process-wide pass registry shim. Named on the `hc` submodule for
  // symmetry with `register_dialects`; only registers hc's own pass
  // families. Upstream stock passes (transform-interpreter, canonicalize,
  // ...) are already registered by `_mlirRegisterEverything` during
  // `_site_initialize`, so callers can freely reference them in pipeline
  // strings without calling anything here. The CAPI side is idempotent —
  // calling this once per context in the same process is safe.
  hcM.def("register_passes", []() { mlirRegisterHCAllPasses(); });
}
