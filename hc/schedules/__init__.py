# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Transform-dialect schedules driving hc's compilation pipeline.

Each `.mlir` file here is a module carrying a `transform.named_sequence
@__transform_main` that invokes registered MLIR passes in order. The
Python driver loads one via `-transform-preload-library`, then runs it
with `-transform-interpreter`. Shipping schedules as IR (rather than a
pass-pipeline string) keeps them overrideable by users without touching
Python and lets the same schedule be reused from `hc-opt` CLI runs.
"""
