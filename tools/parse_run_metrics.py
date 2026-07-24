#!/usr/bin/env python3
"""Parse CosyVoice2 concurrent run logs using the README acceptance rules."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

YIELD_RE = re.compile(r"\[INFO\] yield speech len ([0-9.]+), rtf ([0-9.]+)")
INFER_RE = re.compile(r"\[INFO\] infer round")


def parse_client_log(text: str) -> tuple[list[float], list[float]]:
    idx = text.find("[INFO] infer round")
    if idx < 0:
        return [], []
    first_ms: list[float] = []
    middle_rtf: list[float] = []
    current: list[tuple[float, float]] = []
    for line in text[idx:].splitlines():
        if INFER_RE.search(line):
            if current:
                sl, rtf = current[0]
                first_ms.append(sl * rtf * 1000.0)
                for sl_mid, rtf_mid in current[1:-1]:
                    middle_rtf.append(rtf_mid)
            current = []
            continue
        match = YIELD_RE.search(line)
        if match:
            current.append((float(match.group(1)), float(match.group(2))))
    if current:
        sl, rtf = current[0]
        first_ms.append(sl * rtf * 1000.0)
        for _, rtf_mid in current[1:-1]:
            middle_rtf.append(rtf_mid)
    return first_ms, middle_rtf


def parse_run_dir(run_dir: Path) -> tuple[list[float], list[float]]:
    all_first: list[float] = []
    all_middle: list[float] = []
    for log_path in sorted(run_dir.glob("client_*.log")):
        first, middle = parse_client_log(log_path.read_text(errors="replace"))
        all_first.extend(first)
        all_middle.extend(middle)
    return all_first, all_middle


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory containing client_*.log")
    args = parser.parse_args()

    first_ms, middle_rtf = parse_run_dir(args.run_dir)
    if not first_ms or not middle_rtf:
        raise SystemExit(f"no formal infer metrics found under {args.run_dir}")

    gt03 = 100.0 * float(np.mean(np.array(middle_rtf) > 0.3))
    print(f"run_dir={args.run_dir}")
    print(f"first_count={len(first_ms)} middle_count={len(middle_rtf)}")
    print(f"first_p90_ms={percentile(first_ms, 90):.3f}")
    print(f"first_p95_ms={percentile(first_ms, 95):.3f}")
    print(f"middle_avg_rtf={float(np.mean(middle_rtf)):.6f}")
    print(f"middle_p90_rtf={percentile(middle_rtf, 90):.6f}")
    print(f"middle_p95_rtf={percentile(middle_rtf, 95):.6f}")
    print(f"middle_rtf_gt_0_3_pct={gt03:.2f}%")


if __name__ == "__main__":
    main()
