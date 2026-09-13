"""Runs all 5 audit modules in sequence and prints a one-line summary of each.

Run: uv run python -m redteam.run_all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam import (  # noqa: E402
    m1_answer_leakage,
    m2_tool_poisoning,
    m3_path_traversal,
    m4_lethal_trifecta,
    m5_tool_bloat,
)

MODULES = [
    ("1. answer leakage", m1_answer_leakage),
    ("2. tool description poisoning", m2_tool_poisoning),
    ("3. path traversal", m3_path_traversal),
    ("4. lethal trifecta", m4_lethal_trifecta),
    ("5. tool bloat", m5_tool_bloat),
]


def main() -> None:
    for label, module in MODULES:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        module.main()


if __name__ == "__main__":
    main()
