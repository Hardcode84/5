# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# No tablegen ops module — `hc` is built via conversion + transform,
# not op-by-op from Python. Shim lets callers
# `from hc_mlir.dialects import hc; hc.register_dialects(ctx)` without
# reaching into `_mlir_libs`.
from .._mlir_libs._hcFrontDialectsNanobind.hc import *  # noqa: F403
