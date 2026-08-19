"""
Record the golden replay.

Run this ONLY when a number is expected to change, and put the reason and the
before/after in the commit message. A re-recording with no explanation is how a
golden test stops being evidence.

    uv run python tests/record_golden.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_golden_replay import GOLDEN, observed  # noqa: E402


def main() -> int:
    fresh = observed(Path(tempfile.mkdtemp()))
    previous = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else None

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")

    if previous is None:
        print(f"Recorded {GOLDEN} for the first time.")
        return 0

    changed = [k for k in fresh if previous.get(k) != fresh[k]]
    if not changed:
        print(f"{GOLDEN} was already up to date; nothing changed.")
        return 0
    print(f"Re-recorded {GOLDEN}. Changed keys:")
    for key in changed:
        print(f"  {key}\n    was {previous.get(key)}\n    now {fresh[key]}")
    print("\nPut the reason for each change in the commit message.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
