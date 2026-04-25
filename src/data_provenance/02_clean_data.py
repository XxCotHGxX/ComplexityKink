"""
Step 2: Clean and filter the raw stratified sample.

Input:  stratified_sample_100k.jsonl (71,230 records)
Output: cleaned_gold_set_50k.jsonl   (35,499 records)

Cleaning steps:
  1. Extract code from markdown blocks
  2. Syntax validation (Python: compile(); others: marker heuristic)
  3. Minimum code length (20 chars)
  4. Test integrity (unit_tests present + contains 'assert')
  5. Cap each language at 10,000 samples (random_state=42)

KNOWN ISSUES (see data_provenance/README.md):
  - Python gets real syntax validation; other languages get single-marker check
  - The 10k cap uses pandas .sample(random_state=42) for reproducibility

Original location: D:/ProgD/InstructionEntropy_Economics/clean_data.py
Copied here for complete data provenance.
"""
import json
import os
import zlib
import re
import pandas as pd
from typing import Dict, Any, Optional

# CONFIG
INPUT_PATH = "data/stratified_sample_100k.jsonl"
CLEANED_OUTPUT_PATH = "data/cleaned_gold_set_50k.jsonl"
LOG_PATH = "data/cleaning_log.txt"
TARGET_TOTAL = 50000

def get_compression_ratio(instruction: str, output: str) -> float:
    """Calculates E metric using zlib compression ratio."""
    if not instruction or not output:
        return 0.0
    c_instr = len(zlib.compress(instruction.encode('utf-8')))
    c_out = len(zlib.compress(output.encode('utf-8')))
    return c_out / c_instr if c_instr > 0 else 0.0

def clean_code(code: str, lang: str) -> str:
    """Extracts raw code from markdown blocks, preferring the last block for consistency."""
    if "```" in code:
        # Find all blocks (common in reasoning models)
        pattern = r"```(?:\w+)?\n(.*?)\n```"
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            # Return the LAST block as it is more likely the final answer
            return matches[-1].strip()
    return code.strip()

def validate_syntax(code: str, lang: str) -> bool:
    """Performs a basic syntax check for the specific language."""
    if not code or len(code) < 10:
        return False
    
    if lang == 'python':
        try:
            compile(code, '<string>', 'exec')
            return True
        except:
            return False
    
    # KNOWN ISSUE: non-Python languages use weak marker heuristics
    markers = {
        'java': ['class ', '{', '}'],
        'javascript': ['function', 'const', 'let', 'var', '=>', '{'],
        'cpp': ['#include', 'main', ';', '{'],
        'go': ['package ', 'func ', '{']
    }
    
    found_markers = [m for m in markers.get(lang, []) if m in code]
    return len(found_markers) >= 1

def main():
    print("Starting Data Cleaning Phase...")
    
    stats = {
        'total_seen': 0,
        'removed_syntax': 0,
        'removed_short': 0,
        'removed_no_tests': 0,
        'final_counts': {}
    }
    
    cleaned_samples = []
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file {INPUT_PATH} not found.")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            stats['total_seen'] += 1
            try:
                sample = json.loads(line)
            except:
                continue
                
            lang = sample.get('lang')
            raw_output = sample.get('output', '')
            unit_tests = sample.get('unit_tests', '')
            
            # 1. Code Extraction
            code = clean_code(raw_output, lang)
            
            # 2. Syntax Guard
            if not validate_syntax(code, lang):
                stats['removed_syntax'] += 1
                continue
                
            # 3. Minimum Viable Logic Guard
            if len(code) < 20:
                stats['removed_short'] += 1
                continue
                
            # 4. Test Integrity Guard
            if not unit_tests or len(unit_tests) < 10 or 'assert' not in unit_tests.lower():
                stats['removed_no_tests'] += 1
                continue
            
            sample['code_cleaned'] = code
            sample['e_metric_refined'] = get_compression_ratio(sample['input'], code)
            
            cleaned_samples.append(sample)
            stats['final_counts'][lang] = stats['final_counts'].get(lang, 0) + 1
            
            if len(cleaned_samples) % 5000 == 0:
                print(f"Processed {stats['total_seen']} samples. Cleaned: {len(cleaned_samples)}")

    # Cap at 10k per language for balance
    df = pd.DataFrame(cleaned_samples)
    final_gold_set = []
    
    for lang in ['python', 'javascript', 'go', 'java', 'cpp']:
        lang_group = df[df['lang'] == lang]
        take_n = min(len(lang_group), 10000)
        final_gold_set.extend(lang_group.sample(n=take_n, random_state=42).to_dict('records'))
    
    print(f"Final Gold Set Size: {len(final_gold_set)}")
    
    with open(CLEANED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for s in final_gold_set:
            f.write(json.dumps(s) + '\n')
            
    with open(LOG_PATH, 'w') as f:
        f.write(f"CLEANING REPORT\n{'='*20}\n")
        f.write(json.dumps(stats, indent=2))
        f.write(f"\nFinal Set Language Distribution:\n")
        for s in final_gold_set:
            stats['final_dist'] = stats.get('final_dist', {})
            stats['final_dist'][s['lang']] = stats['final_dist'].get(s['lang'], 0) + 1
        f.write(json.dumps(stats['final_dist'], indent=2))

    print(f"Cleaning Complete. Report saved to {LOG_PATH}")

if __name__ == "__main__":
    main()
