"""Compare two bench.py result files.

Usage:  uv run python benchmarks/compare.py benchmarks/results/baseline.json benchmarks/results/after.json
"""

import json
import sys
from pathlib import Path


def median(xs) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else round((xs[mid - 1] + xs[mid]) / 2, 3)


def metric(entry: dict, key: str) -> float | None:
    return median([r.get(key) for r in entry.get("runs", [])])


def fmt(x: float | None) -> str:
    return f"{x:5.2f}s" if x is not None else "    --"


def delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    pct = (d / a * 100) if a else 0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}s ({sign}{pct:.0f}%)"


def main() -> None:
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())

    print(f"A = {a['label']} ({a.get('git_rev')}, {a.get('timestamp')})")
    print(f"B = {b['label']} ({b.get('git_rev')}, {b.get('timestamp')})")

    for key, title in [("t_first_audio", "time to first audio"), ("t_total", "total turn time")]:
        print(f"\n== {title} ==")
        print(f"{'test':<20} {'A':>7} {'B':>7}  delta")
        for name in a["perf"]:
            if name not in b["perf"]:
                continue
            ma, mb = metric(a["perf"][name], key), metric(b["perf"][name], key)
            print(f"{name:<20} {fmt(ma):>7} {fmt(mb):>7}  {delta(ma, mb)}")


if __name__ == "__main__":
    main()
