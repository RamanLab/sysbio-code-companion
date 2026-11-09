"""Catch a common MATLAB mistake before MATLAB runs.

A MATLAB script cannot use its own name for a variable or a local
function: efms.m may not assign to `efms`, and optknock.m may not define
`function ... = optknock(...)`. Python has no such rule, so this is easy
to miss when translating. Run: python tests/lint_matlab.py
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
problems = []
for path in sorted((root / "examples").rglob("*.m")):
    name = path.stem
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        code = line.split("%", 1)[0]
        if re.match(rf"\s*function\b.*\b{name}\s*\(", code) or \
           re.match(rf"\s*function\s+{name}\s*$", code):
            problems.append(f"{path.relative_to(root)}:{number}: local function "
                            f"named '{name}', like the script")
        elif re.match(rf"\s*(\[[^\]]*\b{name}\b[^\]]*\]|{name})\s*(\(.*\))?\s*=[^=]", code):
            problems.append(f"{path.relative_to(root)}:{number}: variable named "
                            f"'{name}', like the script")
print("\n".join(problems) or "No script/name clashes.")
sys.exit(1 if problems else 0)
