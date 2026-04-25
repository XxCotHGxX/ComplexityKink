"""
Feature Extractor for IV Analysis.
Extracts instruction-based complexity features (instruments) and enriches the dataset.

DATA PROVENANCE: This is the ONLY file that WRITES enriched JSONL records.
Pass-rate computation uses ``data_loader.compute_pass_rate`` ,  the single
source of truth.  Downstream stages read stored values; they never recompute.

Cyclomatic complexity is computed by ``lizard`` ,  a standard, citable static
analysis tool that implements McCabe's CC uniformly across Python, JavaScript,
Java, C++, and Go.  Using one tool for all languages eliminates cross-tool
measurement inconsistency.
"""
import json
import re
import math
import argparse
import sys
import os
import tempfile

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_SCORED_FILE, DEFAULT_ENRICHED_FILE
from data_loader import compute_pass_rate

# lizard: standard multi-language cyclomatic complexity tool
# Supports Python, JavaScript, Java, C/C++, Go (and more)
import lizard


def extract_instruction_features(instruction):
    """
    Extracts structural and complexity features from the instruction text.
    These features serve as Instruments (Z) to predict true target complexity.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        return {
            'inst_tokens': 0, 'inst_if_count': 0, 'inst_loop_count': 0,
            'inst_class_count': 0, 'inst_func_count': 0, 'inst_logic_count': 0,
            'inst_total_structural': 0, 'inst_avg_word_len': 0.0,
            'inst_conditional_count': 0, 'inst_collection_count': 0,
        }

    # 1. Branching Indicators
    n_if = len(re.findall(r'\b(if|when|whether|case|switch)\b', instruction, re.I))
    n_conditional = len(re.findall(r'\b(condition|scenario|situation|otherwise|else)\b', instruction, re.I))
    
    # 2. Iteration Indicators
    n_loop = len(re.findall(r'\b(loop|iterate|for each|while|repeat|mapping|filter)\b', instruction, re.I))
    n_collection = len(re.findall(r'\b(list|array|dictionary|map|set|collection|stream)\b', instruction, re.I))
    
    # 3. Structural Hints (Architectural Complexity)
    n_class = len(re.findall(r'\b(class|object|method|inheritance|interface|abstract)\b', instruction, re.I))
    n_func = len(re.findall(r'\b(function|procedure|recursive|recursion)\b', instruction, re.I))
    n_logic = len(re.findall(r'\b(logic|algorithm|validate|parse|sort|search)\b', instruction, re.I))
    
    # 4. Length/Token Proxies
    tokens = instruction.split()
    n_tokens = len(tokens)
    avg_word_len = sum(len(w) for w in tokens) / n_tokens if n_tokens > 0 else 0
    
    return {
        'inst_tokens': n_tokens,
        'inst_if_count': n_if,
        'inst_conditional_count': n_conditional,
        'inst_loop_count': n_loop,
        'inst_collection_count': n_collection,
        'inst_class_count': n_class,
        'inst_func_count': n_func,
        'inst_logic_count': n_logic,
        'inst_total_structural': n_if + n_loop + n_class + n_func,
        'inst_avg_word_len': avg_word_len,
    }


LANG_TO_LIZARD_EXT = {
    'python': '.py',
    'javascript': '.js',
    'java': '.java',
    'cpp': '.cpp',
    'go': '.go',
}


def compute_cyclomatic_complexity(code, lang):
    """
    Compute McCabe's cyclomatic complexity using lizard.

    Returns the sum of CC across all functions in the code snippet, or
    None if the code is empty, unparsable, or the language is unsupported.

    Using lizard (a standard, citable tool) ensures a consistent CC
    definition across all 5 languages.  This eliminates the risk of
    custom per-language implementations measuring different things.
    """
    if not code or not isinstance(code, str) or len(code.strip()) < 10:
        return None

    ext = LANG_TO_LIZARD_EXT.get(lang)
    if ext is None:
        return None

    try:
        analysis = lizard.analyze_file.analyze_source_code(f"snippet{ext}", code)
        if not analysis.function_list:
            # No functions detected ,  could be a script-level snippet.
            # lizard can't compute CC without function boundaries.
            return None
        total_cc = sum(f.cyclomatic_complexity for f in analysis.function_list)
        return total_cc if total_cc > 0 else None
    except Exception:
        return None


def process_batch_for_iv(input_file, output_file):
    print(f"Loading data from {input_file}...")
    
    processed_count = 0
    cc_computed = 0
    cc_failed = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            
            # Extract instruction-based features (instruments)
            features = extract_instruction_features(data.get('input', ''))
            data['iv_features'] = features
            
            # Compute continuous pass rate via canonical function
            pass_rate = compute_pass_rate(data.get('status', data.get('tests_execution_status', [])))
            data['pass_rate'] = pass_rate
            
            # Compute cyclomatic complexity via lizard (all 5 languages)
            # Use 'code_cleaned' if available, fall back to 'output'
            lang = data.get('lang', 'unknown')
            code = data.get('code_cleaned', data.get('output', ''))
            cc = compute_cyclomatic_complexity(code, lang)
            data['kappa_cyclomatic'] = cc
            if cc is not None:
                cc_computed += 1
            else:
                cc_failed += 1
            
            data.setdefault('coupling_depth', 0)
            data.setdefault('lang', 'unknown')
            
            f_out.write(json.dumps(data) + '\n')
            processed_count += 1
            if processed_count % 50000 == 0:
                print(f"Processed {processed_count} samples "
                      f"(CC: {cc_computed} ok, {cc_failed} None)...")

    print(f"\nDone. Processed {processed_count} total.")
    print(f"  CC computed: {cc_computed} ({cc_computed/max(processed_count,1)*100:.1f}%)")
    print(f"  CC failed:   {cc_failed} ({cc_failed/max(processed_count,1)*100:.1f}%)")
    print(f"  Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract IV features from scored dataset")
    parser.add_argument("--input", default=DEFAULT_SCORED_FILE,
                        help="Path to input scored JSONL file")
    parser.add_argument("--output", default=DEFAULT_ENRICHED_FILE,
                        help="Path to output enriched JSONL file")
    args = parser.parse_args()
    
    process_batch_for_iv(args.input, args.output)
