# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# No tablegen-generated ops module today — the `hc` dialect is targeted by
# the conversion pass and driven through a transform schedule rather than
# constructed op-by-op from Python. This shim exists so callers can say
# `from hc_mlir.dialects import hc; hc.register_dialects(ctx)` without
# reaching into `_mlir_libs`.
from .._mlir_libs._hcFrontDialectsNanobind.hc import *  # noqa: F403
