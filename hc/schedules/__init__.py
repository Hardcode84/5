# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Transform-dialect schedules for hc's compile pipeline.

Each `.mlir` carries a `transform.named_sequence @__transform_main`.
Driver loads via `-transform-preload-library` + `-transform-interpreter`.
Overrideable; reusable from `hc-opt`.
"""
