# Complexity Kink Research

**Identifying a Hard Reliability Threshold in LLM Code Generation via Instrumental Variables**

## Overview

Current LLM code generation benchmarks measure complexity from model *outputs*, not task *inputs*. When a model fails to generate complex code, the broken output registers as "simple" (low cyclomatic complexity). This endogeneity bias hides a structural break in the complexity-performance relationship.

Using Two-Stage Least Squares (2SLS) on N=35,499 coding tasks, we find:

| Finding | Value |
|---------|-------|
| Complexity Kink threshold | κ ≈ 6.5 |
| Pass rate below kink | 40.4% |
| Pass rate above kink | 11.8% |
| First-stage F-statistic | 1,765 |
| Hausman endogeneity test | χ² = 108.5, p < 0.001 |
| Hansen bootstrap p-value | < 0.001 |

## Chain of Evidence (Statistical Provenance)

The findings in this paper are derived from a verifiable chain of statistical tests. Any researcher can reproduce these results using the provided pipeline:

1. **Endogeneity Confirmation:** Hausman test confirms output-based complexity is endogenous ($p < 0.001$), necessitating the IV approach.
2. **Instrument Strength:** First-stage $F$-statistic of **1,765** exceeds standard strength requirements by over 170x.
3. **Threshold Selection:** Bootstrap Hansen test identifies the optimal structural break at $\kappa \approx 6.5$ ($p < 0.001$).
4. **Robustness:** 500-iteration placebo tests confirm the "Complexity Kink" is not a statistical artifact of the instrument construction.

For a full reproducibility audit, see [data_provenance_report.md](paper/data_provenance_report.md).

## Repository Structure


```
├── paper/
│   └── complexity_kink_2026.tex    # Full paper (LaTeX)
├── src/
│   ├── config.py                   # Central configuration
│   ├── feature_extractor_iv.py     # Feature extraction + pass rate computation
│   ├── train_stage1_iv.py          # Stage 1: predict complexity from instructions
│   ├── run_stage2_iv.py            # Stage 2: 2SLS, Hausman, Hansen, placebo
│   ├── generate_viz.py             # Publication figures
│   ├── verify_pipeline.py          # End-to-end smoke test
│   ├── mem_control.py              # Memorization Jaccard control
│   └── parsers/
│       └── py_parser.py            # Tree-Sitter AST feature extraction
├── output/
│   ├── paradox_vs_kink.png         # Naive vs corrected complexity view
│   ├── performance_phase_diagram.png
│   ├── hansen_wald_curve.png       # Threshold selection curve
│   └── stage2_results_summary.txt
└── data/                           # Not tracked (see Data section)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

The data files (~400MB) are not tracked in this repository. The dataset contains coding tasks across 5 languages: Python, JavaScript, Go, Java, and C++.

**Note:** The feature extractor uses Tree-Sitter for deep AST analysis on Python samples. for other languages, it falls back to regex-based complexity features, which are sufficient for the instrumental variables analysis as the instruments are derived from the natural language instructions.

To reproduce:

1. Download `final_results_scored.jsonl` from [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) or generate via the evaluation pipeline.
2. Run feature extraction to produce the enriched dataset:
   ```bash
   python src/feature_extractor_iv.py --input data/final_results_scored.jsonl --output data/iv_enriched_dataset.jsonl
   ```

## Reproducing Results

### Full pipeline
```bash
# Stage 1: Train complexity predictor (10-fold CV)
python src/train_stage1_iv.py

# Stage 2: 2SLS estimation + all diagnostics
python src/run_stage2_iv.py

# Generate publication figures
python src/generate_viz.py
```

### Quick verification (2000-sample subset)
```bash
python src/verify_pipeline.py
```

## Key Methods

- **2SLS Estimation**: `linearmodels.IV2SLS` with instruction-derived instruments
- **Hansen Threshold Test**: Bootstrap sup-Wald with 500 wild bootstrap iterations
- **Hausman Test**: OLS vs 2SLS coefficient comparison
- **Placebo Test**: 500 iterations with shuffled instruments
- **Stage 1 CV**: 10-fold cross-validation with out-of-fold predictions

## Citation

```bibtex
@article{hernandez2026complexity,
  title={The Complexity Kink: Identifying a Hard Reliability Threshold 
         in LLM Code Generation via Instrumental Variables},
  author={Hernandez, Michael},
  year={2026}
}
```

## License



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 [Plethora Solutions, LLC](https://plethorasolutions.llc).

