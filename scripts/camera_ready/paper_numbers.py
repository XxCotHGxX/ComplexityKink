"""Print every outcome-dependent number the manuscript cites, for one result set.

    python scripts/camera_ready/paper_numbers.py results/camera_ready/independent_audit
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    j = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    comb = j("analysis_summary.json")["_combined"]
    bp, sf, ms, iv, oc = (j("breakpoints.json"), j("source_frame_sensitivity.json"),
                          j("model_specific_source_frame.json"), j("iv_diagnostics.json"),
                          j("output_cc_diagnostics.json"))
    rob = j("robustness_summary.json")
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731

    print("== headline")
    print(f"gamma {comb['kink_threshold']} supW {comb['kink_sup_wald']:.2f} p {comb['kink_pval']} "
          f"placebo p {comb['placebo_pval']} CI [{comb['kink_ci_lower']}, {comb['kink_ci_upper']}]")
    print(f"pass low {pct(comb['mean_pass_low'])} high {pct(comb['mean_pass_high'])} "
          f"gap {100 * (comb['mean_pass_high'] - comb['mean_pass_low']):.1f} "
          f"n_low {bp['pool_mean']['n_low']} n_high {bp['pool_mean']['n_high']}")
    print(f"BIC piecewise {comb['piecewise_bic']:.0f} linear {comb['linear_bic']:.0f} cubic {comb['cubic_bic']:.0f}")
    print(f"fprobit marginal {comb.get('fprobit_marginal_at_mean')} (se {comb.get('fprobit_se')}, p {comb.get('fprobit_pval')})")
    print(f"OLS on composite coef {comb.get('ols_coef')} p {comb.get('ols_pval')} r2 {comb.get('ols_r2')}")

    print("== pooling / directions")
    med = bp["pool_median"]
    print(f"median {med['threshold']} low {med['mean_pass_low']:.3f} high {med['mean_pass_high']:.3f} "
          f"gap {100 * (med['mean_pass_high'] - med['mean_pass_low']):.1f}; LOMO {bp['lomo_threshold_range']}")
    print(f"directions {bp['direction_tally']}; model thresholds {bp['per_model_threshold_range']} "
          f"median {bp['per_model_threshold_median']}")
    for r in bp["per_model"]:
        if r["direction"] == "down":
            print(f"  down: {r['model']} gamma {r['threshold']} low {r['mean_pass_low']:.3f} high {r['mean_pass_high']:.3f}")

    print("== task type")
    tt = bp["task_type_controls"]
    fe = tt["fixed_effects"]
    print(f"gamma {fe['threshold']} supW {fe['sup_wald']:.1f} low {fe['mean_pass_low']:.3f} high {fe['mean_pass_high']:.3f} "
          f"gap {100 * fe['raw_regime_gap']:.1f} exc {fe['wild_bootstrap']['exceedances']}/{fe['wild_bootstrap']['draws']}")
    print("R2", {k: round(v, 5) for k, v in tt["piecewise_predictive_fit"].items()})

    print("== construction frame")
    sc, ua = sf["source_controlled_threshold"], sf["unadjusted_threshold"]
    print(f"controlled gamma {sc['threshold']} supW {sc['sup_wald']:.2f} low {sc['mean_pass_low']:.3f} "
          f"high {sc['mean_pass_high']:.3f} gap {100 * sc['raw_regime_gap']:.1f} exc {sc['wild_bootstrap']['exceedances']}")
    for fr, v in sf["within_frame_thresholds"].items():
        print(f"  {fr}: gamma {v['threshold']} supW {v['sup_wald']:.2f} p {v['wild_bootstrap']['p_finite_mc']:.3f} "
              f"low {v['mean_pass_low']:.4f} high {v['mean_pass_high']:.4f} n {v['n_low']}/{v['n_high']}")
    d = sf["source_descriptives"]
    for fr, v in d.items():
        print(f"  {fr}: n {v['n']} mean composite {v['mean_composite']:.2f} mean pass {v['mean_pass']:.3f}")
    fc = sf["fit_comparisons"]
    print("  BIC:", {k: round(v["bic"], 1) for k, v in fc.items()})
    print("  R2:", {k: round(v["r2"], 4) for k, v in fc.items()})
    dec = sf["original_13_75_decomposition"]
    print("  decomposition at 13.75:", json.dumps(dec.get("within_source_cells"))[:400])

    print("== model-specific frame audits")
    for m in ms["models"]:
        name = m.get("display_name", m.get("model_id"))
        un, scm = m["selection_basis_unadjusted_fit"], m["source_controlled_threshold"]
        print(f"  {name}: unadjusted gamma {un['threshold']} low {un['mean_pass_low']:.3f} high {un['mean_pass_high']:.3f}; "
              f"frame-controlled gamma {scm['threshold']} supW {scm['sup_wald']:.2f} "
              f"exc {scm['wild_bootstrap']['exceedances']}/{scm['wild_bootstrap']['draws']}")
        for fr, v in m["within_frame_thresholds"].items():
            pb = v.get("pairs_bootstrap_threshold") or {}
            print(f"     {fr}: gamma {v['threshold']} supW {v['sup_wald']:.2f} "
                  f"p {v['wild_bootstrap']['p_finite_mc']:.3f} low {v['mean_pass_low']:.3f} high {v['mean_pass_high']:.3f} "
                  f"CI [{pb.get('ci_lower')}, {pb.get('ci_upper')}] BIC pw {v['piecewise_fit']['bic']:.1f} "
                  f"lin {v['linear_fit']['bic']:.1f}")

    print("== IV / overid")
    print(json.dumps({k: iv[k] for k in iv if k in ("direct_ols", "candidate_iv_2sls", "sargan", "wooldridge")}, indent=0)[:1400])
    print("subsample:", json.dumps(iv.get("subsample_overidentification") or iv.get("subsample"))[:700])
    print("just-identified:", json.dumps(iv.get("just_identified") or iv.get("single_instrument"))[:700])
    print("pcs:", json.dumps(iv.get("principal_components"))[:900])
    print("subsets non-rejecting:", json.dumps(iv.get("subset_search", {}).get("non_rejecting") or iv.get("non_rejecting_subsets"))[:900])
    print("within task:", json.dumps(iv.get("within_task_type"))[:900])

    print("== output CC / reverse threshold")
    print(json.dumps({k: oc[k] for k in oc if k not in ("kappa_map", "library_mention_diagnostic")})[:1500])
    km = oc.get("kappa_map") or {}
    print("kappa map", {k: km.get(k) for k in ("intercept", "slope", "r2", "n_passing")})
    print("mapping at threshold:", json.dumps(oc.get("passing_generation_mapping"))[:300])

    print("== control specifications (additive vs by side; adjusted jumps in points)")
    cs = j("control_specs.json")
    for name, fit in cs["mean_pooled"].items():
        if name.startswith("adjusted"):
            print("  adjusted jumps at", fit["threshold"],
                  {k: round(100 * v["jump"], 2) for k, v in fit.items() if k != "threshold"})
            continue
        adj = fit["adjusted_jump_at_selected"]
        print(f"  {name}: gamma {fit['threshold']} supW {fit['sup_wald']:.2f} "
              f"exc {fit['wild_bootstrap']['exceedances']}/{fit['wild_bootstrap']['draws']} "
              f"raw {100 * fit['raw_regime_gap']:+.2f} adjusted {100 * adj['jump']:+.2f} (p {adj['p_value']:.3g})")
    for model, fits in cs["model_specific"].items():
        print("  ", model, {k: (v["threshold"], round(v["sup_wald"], 2)) for k, v in fits.items()})

    print("== no-response generations treated as missing")
    nr = j("no_response_sensitivity.json")
    print("  counts", nr["no_response_generations"])
    for label, v in nr["variants"].items():
        p = v["mean_pooled"]
        print(f"  {label}: pooled {p['threshold']} {p['mean_pass_low']:.4f}->{p['mean_pass_high']:.4f}; "
              f"down {v['downward_models']}")

    print("== frame composition at the headline breakpoint")
    print(" ", j("frame_decomposition.json"))
    if (out / "auditor_agreement.json").exists():
        print("== inter-auditor agreement (rule cases excluded)")
        ag = j("auditor_agreement.json")
        for name in ("MAI-Thinking-1", "Phi-4-reasoning"):
            x = ag[name]["excluding_rule_cases"]
            print(f"  {name}: n {x['n']} agreement {x['agreement']:.4f} kappa {x['cohen_kappa']:.4f}")

    print("== extension (raw harness both sides)")
    print(json.dumps(rob["high_complexity_extension"]["matched_five_model"])[:600])
    for name in ("tail_extension_replication.csv", "tail_extension_source_split.csv"):
        rows = list(csv.DictReader(open(out / name, encoding="utf-8")))
        print(name, rows[:12])


if __name__ == "__main__":
    main()
