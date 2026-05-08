/* SPDX-FileCopyrightText: 2026 hc contributors */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

/* Linker version script for hc_execution_engine. Hide every symbol
 * except the Python module init function so the embedded LLVM (linked
 * statically into our nanobind module) can't collide with whatever
 * other LLVM-using extension Python loads in the same process. */
{
  global:
    PyInit_hc_execution_engine;
  local:
    *;
};
