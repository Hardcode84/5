// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-c/Front.h"
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
        MlirDialectHandle hcDialect = mlirGetDialectHandle__hc__();
        MlirContext context_ = context.get()->get();
        mlirDialectHandleRegisterDialect(hcDialect, context_);
        if (load) {
          mlirDialectHandleLoadDialect(hcDialect, context_);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  using namespace hc_front;
  PyValueType::bind(hcFrontM);
  PyTypeExprType::bind(hcFrontM);
}
