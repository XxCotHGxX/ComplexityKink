"""
Central configuration for the Complexity Kink Research pipeline.
All paths, hyperparameters, and constants in one place.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

DEFAULT_SCORED_FILE = os.path.join(DATA_DIR, "final_results_scored.jsonl")
DEFAULT_ENRICHED_FILE = os.path.join(DATA_DIR, "iv_enriched_dataset.jsonl")
DEFAULT_OOF_FILE = os.path.join(DATA_DIR, "iv_enriched_oof.jsonl")
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "kappa_predictor_stage1.joblib")

# --- Stage 1 Hyperparameters ---
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
CV_FOLDS = 10

# --- Stage 2 Parameters ---

# The threshold grid is built in one of two modes:
#   "percentile" ,  quantiles of kappa_predicted between the bounds that satisfy
#     MIN_REGIME_SIZE on both sides. This is the default because it adapts to
#     the empirical distribution of kappa and guarantees by construction that
#     every candidate threshold produces a valid split. A fixed grid on a
#     right-skewed kappa distribution silently discards high-end candidates
#     (where the high regime falls below MIN_REGIME_SIZE), which would bias
#     the sup-Wald search toward the centre of the distribution.
#   "fixed" ,  legacy grid on an absolute kappa scale. Retained for replication
#     of earlier runs but not used for primary results.
THRESHOLD_GRID_MODE = "percentile"
THRESHOLD_GRID_N_POINTS = 40         # only used when mode == "percentile"
THRESHOLD_GRID_START = 2.0           # only used when mode == "fixed"
THRESHOLD_GRID_END = 15.0            # only used when mode == "fixed"
THRESHOLD_GRID_STEP = 0.5            # only used when mode == "fixed"

HANSEN_BOOTSTRAP_ITERATIONS = 500
PLACEBO_ITERATIONS = 500

# Threshold CI bootstrap count. 200 leaves ~5 draws in each tail of a 95% CI
# which gives a percentile estimate with >10% Monte-Carlo noise on the endpoint;
# 1000 is the minimum defensible number for a percentile CI reported as a
# paper result (Davidson & MacKinnon 2004, §9.5).
THRESHOLD_CI_BOOTSTRAP = 1000

# Smallest regime the sup-Wald search is allowed to consider. With ~10
# parameters per regime, 500 keeps the Wald F comfortably above the
# degrees-of-freedom floor and stops the grid search from chasing tiny-tail
# artefacts.
MIN_REGIME_SIZE = 500

# --- Clustering ---
# The sample has one row per (prompt_id, model_id) pair. Observations that
# share a prompt_id are not independent: a prompt's latent difficulty drives
# pass_rate and instrument values for every model that answers it. Ignoring
# this clustering would understate standard errors and produce misleadingly
# narrow bootstrap CIs. Every 2SLS fit and every bootstrap resample in Stage 2
# clusters on this column.
CLUSTER_COL = "id"

# --- Feature columns used as instruments ---
IV_FEATURE_COLS = [
    'inst_tokens',
    'inst_if_count',
    'inst_loop_count',
    'inst_class_count',
    'inst_func_count',
    'inst_logic_count',
    'inst_total_structural',
    'inst_avg_word_len',
]

# `inst_tokens` and `inst_avg_word_len` are length/lexical proxies. A longer prompt
# does not monotonically imply a more complex solution (it can also imply heavy
# spec text or examples that *reduce* required work). Monotonicity of the instrument
# in the latent complexity is required for a clean first stage, so we offer a
# reduced set that drops them ,  used in robustness runs to confirm the kink is
# not an artifact of the length/lexical channel.
REDUCED_IV_COLS = [
    'inst_if_count',
    'inst_loop_count',
    'inst_class_count',
    'inst_func_count',
    'inst_logic_count',
    'inst_total_structural',
]

# Conservative set: the four most defensible structural signals. Used in the
# final robustness table ,  if the kink survives with only these four instruments,
# the result is not being carried by weaker proxies (logic/total_structural are
# composites that could smuggle in length-like variation).
CONSERVATIVE_IV_COLS = [
    'inst_if_count',
    'inst_loop_count',
    'inst_class_count',
    'inst_func_count',
]

# --- Control variables for Stage 2 ---
# Deliberately empty. Previous versions of this list contained `e_norm`
# (compression-ratio entropy of the generated code) and `m_mem_jaccard`
# (3-gram Jaccard overlap between the instruction and the generated code).
# Both are computed from the LLM output, which makes them post-treatment
# variables: they are caused by the same latent "did the model handle this
# correctly?" that drives pass_rate. Conditioning on them is the classic
# "bad controls" problem (Angrist & Pischke 2009, §3.2.3) ,  it blocks part
# of the causal path we are trying to estimate and re-introduces the
# endogeneity the IV strategy was designed to eliminate.
#
# Any legitimate control must be derivable from the instruction alone
# (pre-generation) or from fixed task metadata (language, source dataset).
# If you add controls here, document WHY they are exogenous to generation.
CONTROL_COLS: list[str] = []
