"""Construction-frame composition at a given pooled threshold, for one outcome.

    CK_SCORED_DIR=data/stage_d/scored_independent_audit \
        python scripts/camera_ready/frame_decomposition.py --threshold 14.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analyze_source_frame_sensitivity import load_analysis_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()
    frame, _ = load_analysis_frame(args.data_root)
    high = frame["composite"] > args.threshold
    later = frame["selection_source"] == "stage_d_candidate"
    out = {"threshold": args.threshold,
           "later_share_at_or_below": float(later[~high].mean()),
           "later_share_above": float(later[high].mean())}
    for name, mask in (("earlier", ~later), ("later", later)):
        out[name] = {"pass_at_or_below": float(frame.loc[mask & ~high, "pass_rate"].mean()),
                     "pass_above": float(frame.loc[mask & high, "pass_rate"].mean()),
                     "n_at_or_below": int((mask & ~high).sum()), "n_above": int((mask & high).sum())}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
