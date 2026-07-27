"""Generate Stage D paper figures from locked prompt/model outputs.

The script writes the filenames referenced by the NeurIPS draft:
``paper/pipeline.png``, ``paper/complexity_kink.png``,
``paper/heatmap_E_per_model_kink.png``, and ``paper/sankey.png``.
All plotted thresholds and statistics are read from Stage D result files or
recomputed from the scored outputs; no reported result is hand-entered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_kink import RUBRIC_DIMS, build_combined_df, load_rubric_scores, load_scored_model  # noqa: E402

STAGE_D_DIR = ROOT / "data" / "stage_d"
SCORED_DIR = STAGE_D_DIR / "scored_combined"
RUBRIC_PATH = STAGE_D_DIR / "ensemble_scores_current_aggregated.jsonl"
SUMMARY_PATH = ROOT / "results" / "analysis_summary.json"
PER_MODEL_SUMMARY_PATH = ROOT / "results" / "per_model_bootstrap_summary.csv"
PAPER_DIR = ROOT / "paper"

FIG_DPI = 220
BLUE = "#2f6fbb"
GREEN = "#2f8f5b"
RED = "#b83a3a"
ORANGE = "#d6802a"
GRAY = "#666666"
LIGHT_GRAY = "#e6e6e6"

# Display-only bins for the Appendix Sankey. Lizard CC=1 is the
# straight-line/stub-like case central to the Reverse Threshold Problem; the
# remaining CC bands aggregate low, moderate, high, and very high generated
# output complexity for readability. The rubric cutoff at 8 is one third of the
# 24-point scale, used only to color prompts beyond basic intended structure.
OUTPUT_CC_STUB_MAX = 1
OUTPUT_CC_LOW_MAX = 5
OUTPUT_CC_MODERATE_MAX = 10
OUTPUT_CC_HIGH_MAX = 20
RUBRIC_BASIC_STRUCTURE_MAX = 8.0
RUBRIC_LOW_BAND_MAX = 6.0
RUBRIC_MID_BAND_MAX = 10.0
ZERO_PASS_MAX = 0.0
# Dense-support display range for binned pass-rate figures. Rounded bin 0 has
# only 8 prompts, while bins 17, 18, and 19 have only 25, 8, and 3 prompts.
# Sparse bins are retained in all statistical estimates and reported aggregate
# pass rates, even when omitted from visual summaries.
DENSE_MIN_COMPOSITE_BIN = 1
MEAN_KINK_MAX_COMPOSITE_BIN = 17
MEAN_KINK_XMAX = 17.5
HEATMAP_MAX_COMPOSITE_BIN = 16
# Visual-only cutoff: below this approximate relative luminance, white cell
# text is more legible than dark text on the pass-rate heatmap.
HEATMAP_TEXT_LIGHTNESS_CUTOFF = 0.48


DISPLAY_NAMES = {
    "anthropic_claude-opus-4.6": "Claude Opus 4.6",
    "anthropic_claude-opus-4.7": "Claude Opus 4.7",
    "anthropic_claude-sonnet-4.6": "Claude Sonnet 4.6",
    "arcee-ai_trinity-large-preview_free": "Trinity-large",
    "azure_deepseek-v3.2-speciale": "DeepSeek V3.2",
    "azure_gpt-oss-120b": "GPT-OSS-120B",
    "azure_grok-3": "Grok-3",
    "azure_kimi-k2.5": "Kimi K2.5",
    "azure_llama-3.3-70b": "Llama 3.3-70B",
    "azure_mistral-large-3": "Mistral Large-3",
    "glm_4_7_flash_results": "GLM 4.7-flash",
    "google_gemini-3-flash-preview": "Gemini 3 Flash",
    "google_gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-4.1": "GPT-4.1",
    "gpt-5-mini": "GPT-5-mini",
    "gpt-oss-20b": "GPT-OSS-20B",
    "ministral-3-14b-reasoning": "Ministral-3-14B",
    "mistral-small-2412": "Mistral Small 2412",
    "openai_gpt-5.4": "GPT-5.4",
    "qwen3.5-9b": "Qwen 3.5-9B",
    "qwen_qwen3.6-plus": "Qwen 3.6 Plus",
}


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["_combined"]


def load_model_frames() -> dict[str, pd.DataFrame]:
    rubric = load_rubric_scores(RUBRIC_PATH)
    frames = {}
    for path in sorted(SCORED_DIR.glob("*.jsonl")):
        frames[path.stem] = load_scored_model(path, rubric)
    return frames


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_pipeline() -> None:
    stages = [
        ("OpenCodeInstruct", "5,000 stratified\nPython prompts"),
        ("Judge Ensemble", "4 out-of-panel LLMs\n6 rubric dimensions"),
        ("Model Panel", "21 models x 5,000\ngenerations"),
        ("Measurements", "prompt complexity,\noutput CC, pass rate"),
        ("Inference", "2SLS diagnostics and\nHansen threshold tests"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 2.8))
    ax.axis("off")
    y = 0.55
    box_w = 0.152
    xs = np.linspace(0.095, 0.905, len(stages))
    for i, (x, (title, body)) in enumerate(zip(xs, stages)):
        rect = plt.Rectangle(
            (x - box_w / 2, y - 0.22),
            box_w,
            0.44,
            facecolor="#f7f9fb",
            edgecolor=BLUE,
            linewidth=1.4,
            joinstyle="round",
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.065, title, ha="center", va="center", weight="bold", color="#1f2937", fontsize=10)
        ax.text(x, y - 0.075, body, ha="center", va="center", color="#374151", linespacing=1.25, fontsize=9)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - box_w / 2 - 0.012, y),
                xytext=(x + box_w / 2 + 0.012, y),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4),
            )
    ax.text(
        0.5,
        0.09,
        "Prompt-side complexity is measured before generation; output complexity is retained only as a comparison measure.",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "pipeline.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_complexity_kink(combined: pd.DataFrame, summary: dict) -> None:
    gamma = float(summary["kink_threshold"])
    ci_low = float(summary["kink_ci_lower"])
    ci_high = float(summary["kink_ci_upper"])
    binned = (
        combined.assign(composite_bin=combined["composite"].round().astype(int))
        .groupby("composite_bin", observed=True)
        .agg(
            n=("pass_rate", "size"),
            pass_rate=("pass_rate", "mean"),
            se=("pass_rate", "sem"),
            output_cc=("kappa_cyclomatic", "mean"),
        )
        .reset_index()
    )
    dense_binned = binned[
        binned["composite_bin"].between(DENSE_MIN_COMPOSITE_BIN, MEAN_KINK_MAX_COMPOSITE_BIN)
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.plot(dense_binned["composite_bin"], dense_binned["pass_rate"], color=BLUE, marker="o", lw=2)
    ax.fill_between(
        dense_binned["composite_bin"],
        dense_binned["pass_rate"] - 1.96 * dense_binned["se"].fillna(0),
        dense_binned["pass_rate"] + 1.96 * dense_binned["se"].fillna(0),
        color=BLUE,
        alpha=0.14,
        linewidth=0,
    )
    ax.axvline(gamma, color=RED, linestyle="--", lw=1.6)
    ax.axvspan(ci_low, ci_high, color=RED, alpha=0.08)
    ax.text(gamma + 0.15, 0.965, fr"$\hat{{\gamma}}={gamma:.2f}$", color=RED, va="top")
    dense_mid = dense_binned[dense_binned["composite_bin"] <= gamma]
    trough = dense_mid.loc[dense_mid["pass_rate"].idxmin()]
    ax.scatter([trough["composite_bin"]], [trough["pass_rate"]], color=ORANGE, zorder=5)
    ax.annotate(
        "mid-complexity trough",
        xy=(trough["composite_bin"], trough["pass_rate"]),
        xytext=(trough["composite_bin"] + 1.1, trough["pass_rate"] - 0.045),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1),
        color="#7a4b11",
    )
    ax.text(
        0.02,
        0.06,
        f"Below threshold: {summary['mean_pass_low']:.1%}\nAbove threshold: {summary['mean_pass_high']:.1%}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=LIGHT_GRAY),
    )
    ax.set_xlabel("Rubric composite score")
    ax.set_ylabel("Mean pass rate across 21 models")
    ax.set_ylim(0.45, 0.98)
    ax.set_xlim(DENSE_MIN_COMPOSITE_BIN - 0.5, MEAN_KINK_XMAX)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "complexity_kink.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(model_frames: dict[str, pd.DataFrame], per_model: pd.DataFrame, summary: dict) -> None:
    del summary
    bins = list(range(DENSE_MIN_COMPOSITE_BIN, HEATMAP_MAX_COMPOSITE_BIN + 1))
    threshold_by_model = {
        row["model_id"]: float(row["kink_threshold"])
        for _, row in per_model.iterrows()
    }
    ordered_models = sorted(
        threshold_by_model,
        key=lambda model_id: (threshold_by_model[model_id], DISPLAY_NAMES.get(model_id, model_id)),
    )

    rows = []
    labels = []
    for model_id in ordered_models:
        df = model_frames[model_id]
        working = df[["pass_rate", "composite"]].dropna().copy()
        working["composite_bin"] = working["composite"].round().astype(int)
        means = working.groupby("composite_bin", observed=True)["pass_rate"].mean().reindex(bins)
        rows.append(pd.Series(means, index=bins, name=model_id))
        labels.append(DISPLAY_NAMES.get(model_id, model_id))
    heat = pd.DataFrame(rows, index=labels)

    fig, ax = plt.subplots(figsize=(11.0, 8.3))
    arr = heat.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(col) for col in heat.columns])
    ax.set_xlabel("Rounded ensemble rubric composite (prompt complexity)")
    ax.set_title("Each model's own kink (diamond) by prompt complexity", fontsize=18, weight="bold", pad=10)

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            val = arr[y, x]
            if np.isnan(val):
                label = "-"
                color = "#1f2937"
            else:
                label = f"{val:.0%}"
                rgba = im.cmap(im.norm(val))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < HEATMAP_TEXT_LIGHTNESS_CUTOFF else "#111827"
            ax.text(x, y, label, ha="center", va="center", fontsize=5.6, weight="bold", color=color)

    for y, model_id in enumerate(ordered_models):
        gamma = threshold_by_model[model_id]
        marker_x = int(round(gamma)) - DENSE_MIN_COMPOSITE_BIN
        if 0 <= marker_x < len(bins):
            ax.scatter(
                marker_x,
                y,
                marker="D",
                s=110,
                facecolor="#315a2f",
                edgecolor="none",
                alpha=0.55,
                zorder=3,
            )

    ax.set_xticks(np.arange(-0.5, len(heat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean pass rate")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "heatmap_E_per_model_kink.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def bin_output_cc(cc: float) -> str:
    if pd.isna(cc):
        return "missing"
    if cc <= OUTPUT_CC_STUB_MAX:
        return "Lizard CC = 1"
    if cc <= OUTPUT_CC_LOW_MAX:
        return "Lizard CC 2-5"
    if cc <= OUTPUT_CC_MODERATE_MAX:
        return "Lizard CC 6-10"
    if cc <= OUTPUT_CC_HIGH_MAX:
        return "Lizard CC 11-20"
    return "Lizard CC 21+"


def bin_rubric(score: float, gamma: float) -> str:
    if score <= RUBRIC_LOW_BAND_MAX:
        return "Rubric 0-6"
    if score <= RUBRIC_BASIC_STRUCTURE_MAX:
        return "Rubric >6-8"
    if score <= RUBRIC_MID_BAND_MAX:
        return "Rubric >8-10"
    if score <= gamma:
        return f"Rubric >10-{gamma:g}"
    return f"Rubric >{gamma:g}"


def save_sankey(model_frames: dict[str, pd.DataFrame], summary: dict) -> None:
    gamma = float(summary["kink_threshold"])
    all_rows = pd.concat(model_frames.values(), ignore_index=True)
    zero_pass = all_rows[all_rows["pass_rate"] <= ZERO_PASS_MAX].copy()
    flow_df = zero_pass[["kappa_cyclomatic", "composite"]].dropna().copy()
    flow_df["source"] = flow_df["kappa_cyclomatic"].map(bin_output_cc)
    flow_df["target"] = flow_df["composite"].map(lambda x: bin_rubric(float(x), gamma))
    flows = flow_df.groupby(["source", "target"], observed=True).size().reset_index(name="count")

    sources = ["Lizard CC = 1", "Lizard CC 2-5", "Lizard CC 6-10", "Lizard CC 11-20", "Lizard CC 21+"]
    targets = ["Rubric 0-6", "Rubric >6-8", "Rubric >8-10", f"Rubric >10-{gamma:g}", f"Rubric >{gamma:g}"]
    labels = sources + targets
    idx = {label: i for i, label in enumerate(labels)}
    link_colors = []
    for _, row in flows.iterrows():
        target_high = row["target"] in {"Rubric >8-10", f"Rubric >10-{gamma:g}", f"Rubric >{gamma:g}"}
        direct_misclass = row["source"] == "Lizard CC = 1" and target_high
        low_output_high_prompt = row["source"] in {"Lizard CC = 1", "Lizard CC 2-5", "Lizard CC 6-10"} and target_high
        if direct_misclass:
            link_colors.append("rgba(184, 58, 58, 0.70)")
        elif low_output_high_prompt:
            link_colors.append("rgba(214, 128, 42, 0.45)")
        else:
            link_colors.append("rgba(47, 111, 187, 0.22)")

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=20,
                thickness=18,
                label=labels,
                color=["#d9e8f6"] * len(sources) + ["#dff1e8"] * len(targets),
                line=dict(color="#777777", width=0.5),
                x=[0.02] * len(sources) + [0.80] * len(targets),
                y=[0.02, 0.20, 0.40, 0.62, 0.82, 0.05, 0.24, 0.43, 0.62, 0.82],
            ),
            link=dict(
                source=[idx[s] for s in flows["source"]],
                target=[idx[t] for t in flows["target"]],
                value=flows["count"],
                color=link_colors,
                hovertemplate="%{source.label} -> %{target.label}<br>%{value:,} zero-pass generations<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        width=1100,
        height=600,
        margin=dict(l=35, r=210, t=85, b=35),
        font=dict(size=14, color="#1f2937"),
        title=dict(
            text="Zero-pass generations: output CC can understate prompt complexity",
            x=0.5,
            xanchor="center",
            font=dict(size=22),
        ),
        annotations=[
            dict(x=0.02, y=1.08, xref="paper", yref="paper", text="Lizard CC on generated output", showarrow=False, font=dict(size=16, color="#1f2937")),
            dict(x=0.80, y=1.08, xref="paper", yref="paper", text="Ensemble rubric composite on prompt", showarrow=False, font=dict(size=16, color="#1f2937")),
            dict(
                x=0.5,
                y=-0.03,
                xref="paper",
                yref="paper",
                text=(
                    f"Filtered to generations passing no unit tests (n={len(flow_df):,}). "
                    "Red/orange flows have Lizard CC <= 10 but prompt rubric > 8 before generation."
                ),
                showarrow=False,
                font=dict(size=12, color="#5b6472"),
            ),
        ],
        paper_bgcolor="white",
    )
    fig.write_image(PAPER_DIR / "sankey.png", scale=2)


def main() -> None:
    style_matplotlib()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    model_frames = load_model_frames()
    combined = build_combined_df(model_frames)
    per_model = pd.read_csv(PER_MODEL_SUMMARY_PATH)

    save_pipeline()
    save_complexity_kink(combined, summary)
    save_heatmap(model_frames, per_model, summary)
    save_sankey(model_frames, summary)

    print("Wrote paper figures:")
    for name in ["pipeline.png", "complexity_kink.png", "heatmap_E_per_model_kink.png", "sankey.png"]:
        path = PAPER_DIR / name
        print(f"  {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
