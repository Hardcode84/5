# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- Python -*-
# ruff: noqa: F821

from pathlib import Path

import lit.formats
from lit.llvm import llvm_config

config.name = "HC"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = str(Path(__file__).parent)
config.test_exec_root = str(Path(config.hc_obj_root) / "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_environment("PATH", config.hc_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.hc_install_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.hc_tools_dir,
    config.hc_install_tools_dir,
    config.llvm_tools_dir,
]
tools = ["FileCheck", "count", "not", "hc-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
