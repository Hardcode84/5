# SPDX-FileCopyrightText: 2026 hc contributors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Reject source files containing non-ASCII bytes. Reports the first
# offender per file as `path:line:col: 0xNN`. Exit non-zero on any hit.

from __future__ import annotations

import sys
from pathlib import Path


def scan(path: Path) -> tuple[int, int, int] | None:
    data = path.read_bytes()
    line = 1
    col = 1
    for byte in data:
        if byte >= 0x80:
            return (line, col, byte)
        if byte == 0x0A:
            line += 1
            col = 1
        else:
            col += 1
    return None


def main(argv: list[str]) -> int:
    failed = False
    for arg in argv[1:]:
        path = Path(arg)
        try:
            offender = scan(path)
        except OSError as exc:
            print(f"{path}: read error: {exc}", file=sys.stderr)
            failed = True
            continue
        if offender is None:
            continue
        line, col, byte = offender
        print(f"{path}:{line}:{col}: non-ASCII byte 0x{byte:02x}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
