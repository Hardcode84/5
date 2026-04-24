# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

KERNEL_RETURN_NONE_SOURCE = """
@kernel(work_shape=(4,), group_shape=(4,))
def demo(group):
    return None
"""

FUNC_RETURN_NONE_SOURCE = """
@kernel.func(scope=WorkGroup)
def helper(group):
    return None
"""
