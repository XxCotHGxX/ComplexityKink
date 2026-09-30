"""Breakpoint searches under alternative control specifications, for one outcome.

Two ways of adding controls (task-type indicators, the construction-frame
indicator) to the sup-Wald breakpoint search:

* ``by_side`` -- the specification of scripts/analyze_source_frame_sensitivity.py
  (``wild_bootstrap_threshold``): the no-break model has one set of control
  coefficients and the break model estimates a separate regression, controls
  included, on each side of gamma. The test therefore also asks whether the
  control effects differ across the break.
* ``additive`` -- the usual fixed-effects specification: the controls enter once,
  and only the composite line may break (a jump and a slope change at gamma).

For each specification it reports the selected breakpoint, sup-Wald statistic,
a 300-draw Rademacher wild bootstrap under the no-break model, the raw regime
means at the selected split, and the covariate-adjusted jump: the coefficient on
1[C > gamma] in an OLS of pass rate on that indicator and the controls (HC1 SE).
Adjusted jumps are also reported at the unadjusted breakpoint, so the raw and
adjusted differences are compared at the same split. The same searches are run
for the model-specific outcomes named by ``--models``.

    CK_SCORED_DIR=data/stage_d/scored_independent_audit python \
        scripts/camera_ready/control_specs.py --out results/camera_ready/independent_audit/control_specs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (HERE, ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analyze_source_frame_sensitivity import _basis, _rss, wild_bootstrap_threshold  # noqa: E402
from breakpoints import load_panel  # noqa: E402
from config import MIN_REGIME_SIZE  # noqa: E402
from run_stage2_iv import build_threshold_grid  # noqa: E402


def adjusted_jump(frame: pd.DataFrame, controls: list[str], gamma: float) -> dict:
    high = (frame["composite"] > gamma).astype(float).rename("above")
    X = sm.add_constant(pd.concat([high, frame[controls]], axis=1), has_constant="add")
    fit = sm.OLS(frame["pass_rate"], X).fit(cov_type="HC1")
    return {"jump": float(fit.params["above"]), "se_hc1": float(fit.bse["above"]),
            "p_value": float(fit.pvalues["above"])}


def additive_threshold(frame: pd.DataFrame, controls: list[str], n_boot: int, seed: int) -> dict:
    """Sup-Wald search with additive controls: y ~ 1 + C + controls + 1[C>g] + (C-g)1[C>g]."""
    x = frame["composite"].to_numpy(dtype=float)
    y = frame["pass_rate"].to_numpy(dtype=float)
    n = len(y)
    base = np.column_stack([np.ones(n), x] + [frame[c].to_numpy(dtype=float) for c in controls])
    q0 = _basis(base)
    candidates = []
    for gamma in build_threshold_grid(frame, "composite"):
        high = (x > float(gamma)).astype(float)
        if n - high.sum() < MIN_REGIME_SIZE or high.sum() < MIN_REGIME_SIZE:
            continue
        q1 = _basis(np.column_stack([base, high, (x - float(gamma)) * high]))
        extra = q1.shape[1] - q0.shape[1]
        if extra < 1:
            continue
        candidates.append((float(gamma), q1, extra))

    def curve(values: np.ndarray) -> list[tuple[float, float]]:
        r0 = _rss(values, q0)
        out = []
        for gamma, q1, extra in candidates:
            r1 = _rss(values, q1)
            if r1 > 0:
                out.append((gamma, ((r0 - r1) / extra) / (r1 / (n - q1.shape[1]))))
        return out

    gamma, sup_wald = max(curve(y), key=lambda item: item[1])
    fitted = q0 @ (q0.T @ y)
    residual = y - fitted
    rng = np.random.RandomState(seed)
    boot = np.asarray([max(v for _, v in curve(fitted + residual * rng.choice([-1.0, 1.0], size=n)))
                       for _ in range(n_boot)])
    exceed = int(np.sum(boot >= sup_wald))
    low = x <= gamma
    return {"threshold": gamma, "sup_wald": float(sup_wald), "n_low": int(low.sum()),
            "n_high": int((~low).sum()), "mean_pass_low": float(y[low].mean()),
            "mean_pass_high": float(y[~low].mean()), "raw_regime_gap": float(y[~low].mean() - y[low].mean()),
            "threshold_grid_candidates": len(candidates),
            "wild_bootstrap": {"seed": seed, "draws": n_boot, "exceedances": exceed,
                               "p_finite_mc": (exceed + 1) / (n_boot + 1),
                               "q95": float(np.quantile(boot, 0.95)), "maximum": float(boot.max())}}


def run_specs(frame: pd.DataFrame, specs: dict[str, tuple[list[str], str]], n_boot: int, seed: int) -> dict:
    out = {}
    for name, (controls, kind) in specs.items():
        if kind == "by_side":
            fit = wild_bootstrap_threshold(frame, controls, n_boot, seed)
        else:
            fit = additive_threshold(frame, controls, n_boot, seed)
        fit["specification"] = kind
        fit["controls"] = list(controls)
        fit["adjusted_jump_at_selected"] = adjusted_jump(frame, controls, fit["threshold"])
        out[name] = fit
    reference = out["unadjusted"]["threshold"]
    out["adjusted_jumps_at_unadjusted_threshold"] = {
        "threshold": reference,
        **{label: adjusted_jump(frame, controls, reference)
           for label, controls in (("none", []), ("task_type", frame.attrs["task"]),
                                   ("frame", ["later"]), ("task_type_and_frame", frame.attrs["task"] + ["later"]))}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=["azure_kimi-k2.5", "google_gemini-3.1-pro-preview"])
    args = ap.parse_args()

    scored_dir = Path(os.environ.get("CK_SCORED_DIR", args.data_root / "stage_d" / "scored_combined"))
    panel, composite = load_panel(args.data_root, scored_dir)
    frames = {}
    for line in open(args.data_root / "stage_d" / "stage_d_prompts.jsonl", encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            frames[rec["prompt_id"]] = rec.get("selection_source")
    meta = pd.read_parquet(args.data_root / "public_release" / "prompts", columns=["prompt_id", "task_type"])
    task = meta.set_index("prompt_id")["task_type"].reindex(panel.index)
    if task.isna().any():
        raise RuntimeError("task-type label missing for some prompts")
    dummies = pd.get_dummies(task, prefix="tt", drop_first=True, dtype=float)
    base = pd.DataFrame({"composite": composite.to_numpy(),
                         "later": (pd.Series(frames).reindex(panel.index) == "stage_d_candidate").astype(float).to_numpy()},
                        index=panel.index)
    base = pd.concat([base, dummies], axis=1).reset_index(drop=True)
    T = list(dummies.columns)

    pooled_specs = {
        "unadjusted": ([], "by_side"),
        "task_type_additive": (T, "additive"),
        "task_type_by_side": (T, "by_side"),
        "frame_additive": (["later"], "additive"),
        "frame_by_side": (["later"], "by_side"),
        "task_type_and_frame_additive": (T + ["later"], "additive"),
    }
    frame = base.copy()
    frame["pass_rate"] = panel.mean(axis=1).to_numpy()
    frame.attrs["task"] = T
    report = {"scored_dir": str(scored_dir), "n_prompts": int(len(frame)),
              "minimum_regime_size": MIN_REGIME_SIZE, "mean_pooled": run_specs(frame, pooled_specs, args.n_boot, args.seed),
              "model_specific": {}}
    model_specs = {"unadjusted": ([], "by_side"), "frame_additive": (["later"], "additive"),
                   "frame_by_side": (["later"], "by_side")}
    for model in args.models:
        mframe = base.copy()
        mframe["pass_rate"] = panel[model].to_numpy()
        mframe.attrs["task"] = T
        fits = {name: {**(wild_bootstrap_threshold(mframe, c, args.n_boot, args.seed) if kind == "by_side"
                          else additive_threshold(mframe, c, args.n_boot, args.seed)), "specification": kind}
                for name, (c, kind) in model_specs.items()}
        report["model_specific"][model] = fits

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, fit in report["mean_pooled"].items():
        if name.startswith("adjusted"):
            continue
        adj = fit["adjusted_jump_at_selected"]
        print(f"{name:32s} gamma={fit['threshold']:5.2f} supW={fit['sup_wald']:7.2f} "
              f"exc={fit['wild_bootstrap']['exceedances']}/{args.n_boot} raw={100 * fit['raw_regime_gap']:+.2f} "
              f"adj={100 * adj['jump']:+.2f} (p={adj['p_value']:.3g})")
    ref = report["mean_pooled"]["adjusted_jumps_at_unadjusted_threshold"]
    print("adjusted jumps at", ref["threshold"], {k: round(100 * v["jump"], 2) for k, v in ref.items() if k != "threshold"})
    for model, fits in report["model_specific"].items():
        print(model, {k: (v["threshold"], round(v["sup_wald"], 2), v["wild_bootstrap"]["exceedances"],
                          round(v["mean_pass_low"], 3), round(v["mean_pass_high"], 3)) for k, v in fits.items()})


if __name__ == "__main__":
    main()
