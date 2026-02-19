"""
Feature Extractor for IV Analysis.
Extracts instruction-based complexity features (instruments) and enriches the dataset.
Integrates Tree-Sitter for Python code when available.
"""
import json
import re
import math
import argparse
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_SCORED_FILE, DEFAULT_ENRICHED_FILE


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


def compute_pass_rate(status_field):
    """
    Computes continuous pass rate from the status field.
    Returns a float in [0, 1].
    """
    if not status_field:
        return 0.0
    if isinstance(status_field, str):
        # Handle stringified list
        try:
            import ast
            status_field = ast.literal_eval(status_field)
        except (ValueError, SyntaxError):
            return 1.0 if 'pass' in status_field.lower() else 0.0
    
    if isinstance(status_field, list):
        if len(status_field) == 0:
            return 0.0
        n_pass = sum(1 for s in status_field if isinstance(s, str) and 'pass' in s.lower())
        return n_pass / len(status_field)
    
    return 0.0


def try_extract_ast_features(code, lang):
    """
    Attempts Tree-Sitter AST feature extraction for Python code.
    Returns empty dict for non-Python or on failure.
    """
    if lang != 'python' or not code or not isinstance(code, str):
        return {}
    
    try:
        from parsers.py_parser import extract_ast_features
        return extract_ast_features(code)
    except Exception:
        return {}


def process_batch_for_iv(input_file, output_file):
    print(f"Loading data from {input_file}...")
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            
            # Extract instruction-based features (instruments)
            features = extract_instruction_features(data.get('input', ''))
            
            # Merge features into the data object
            data['iv_features'] = features
            
            # Compute continuous pass rate (not just binary)
            pass_rate = compute_pass_rate(data.get('status', []))
            data['pass_rate'] = pass_rate
            data['is_success'] = 1 if pass_rate >= 0.8 else 0
            
            # Try AST features for Python code (enriches instruments)
            lang = data.get('lang', '')
            ast_features = try_extract_ast_features(data.get('code_cleaned', ''), lang)
            if ast_features:
                data['ast_features'] = ast_features
            
            # Passthrough fields that already exist (ensure they're present)
            data.setdefault('e_norm', 0.0)
            data.setdefault('m_mem_jaccard', 0.0)
            data.setdefault('coupling_depth', 0)
            data.setdefault('kappa_cyclomatic', 1)
            
            f_out.write(json.dumps(data) + '\n')
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count} samples...")

    print(f"Done. Processed {processed_count} total. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract IV features from scored dataset")
    parser.add_argument("--input", default=DEFAULT_SCORED_FILE,
                        help="Path to input scored JSONL file")
    parser.add_argument("--output", default=DEFAULT_ENRICHED_FILE,
                        help="Path to output enriched JSONL file")
    args = parser.parse_args()
    
    process_batch_for_iv(args.input, args.output)
