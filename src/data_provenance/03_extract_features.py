"""
Step 3: Extract features from cleaned code samples.

Input:  cleaned_gold_set_50k.jsonl    (35,499 records)
Output: feature_extracted_set.jsonl   (35,499 records + features)

Features added per sample:
  - kappa_cyclomatic: McCabe's cyclomatic complexity (radon for Python, lizard for others)
  - coupling_depth:   count of unique import/include statements
  - m_mem_jaccard:    3-gram Jaccard similarity between instruction and code
  - e_norm:           language-normalized compression ratio

KNOWN ISSUES (see data_provenance/README.md):
  - Cyclomatic complexity defaults to 1 on ANY failure (line 29)
  - This conflates "unparsable code" with "trivially simple code"
  - Fixed in the downstream pipeline (parsers/py_parser.py returns None)

Original location: D:/ProgD/InstructionEntropy_Economics/src/extract_features.py
Copied here for complete data provenance.
"""
import json
import os
import zlib
import re
import numpy as np
import pandas as pd
import radon.complexity as cc
from lizard import analyze_file
from typing import Dict, Any, List
from collections import Counter

# CONFIG
INPUT_PATH = "data/cleaned_gold_set_50k.jsonl"
OUTPUT_PATH = "data/feature_extracted_set.jsonl"
LOG_PATH = "data/extraction_errors.log"
STATS_PATH = "data/language_stats.json"

def get_cyclomatic_complexity(code: str, lang: str) -> int:
    """Calculates McCabe's Cyclomatic Complexity.
    
    KNOWN ISSUE: returns 1 on failure, conflating parse errors with simple code.
    """
    try:
        if lang == 'python':
            results = cc.cc_visit(code)
            return sum(item.complexity for item in results) if results else 1
        else:
            analysis = analyze_file.analyze_source_code("snippet." + lang, code)
            return sum(f.cyclomatic_complexity for f in analysis.function_list) if analysis.function_list else 1
    except:
        return 1

def get_dependency_depth(code: str, lang: str) -> int:
    """Estimates structural coupling via unique external references."""
    patterns = {
        'python': r'(?:from\s+[\w\.]+\s+import|import\s+[\w\.,\s]+)',
        'javascript': r'(?:import|require)\s*\(?[\'"]([\w\.\-/]+)[\'"]\)?',
        'java': r'import\s+([\w\.\*]+);',
        'cpp': r'#include\s*[<"]([\w\.]+)[>"]',
        'go': r'import\s+\(?\s*["\']([\w\.\-/]+)["\']'
    }
    pattern = patterns.get(lang)
    if not pattern: return 0
    return len(set(re.findall(pattern, code)))

def get_jaccard_similarity(str1: str, str2: str, n: int = 3) -> float:
    """Calculates n-gram Jaccard similarity between prompt and response."""
    def get_ngrams(text):
        tokens = re.findall(r'\w+', text.lower())
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    a = get_ngrams(str1)
    b = get_ngrams(str2)
    if not a or not b: return 0.0
    return len(a.intersection(b)) / len(a.union(b))

def main():
    print("Starting Feature Extraction...")
    
    # --- PASS 1: COLLECT STATS FOR Z-SCORING ---
    print("Pass 1: Calculating Language-Agnostic Baselines...")
    lang_data = {}
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            lang = sample['lang']
            e_raw = sample.get('e_metric_refined', 0.0)
            
            if lang not in lang_data:
                lang_data[lang] = []
            lang_data[lang].append(e_raw)
    
    stats = {}
    for lang, values in lang_data.items():
        stats[lang] = {
            'mean': np.mean(values),
            'std': np.std(values) if np.std(values) > 0 else 1.0
        }
    
    with open(STATS_PATH, 'w') as f_stats:
        json.dump(stats, f_stats, indent=2)
    print(f"Language stats saved. (Python Mean E: {stats['python']['mean']:.4f})")

    # --- PASS 2: FULL EXTRACTION ---
    print("Pass 2: Extracting Features & Normalizing...")
    processed_count = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out, \
         open(LOG_PATH, 'w', encoding='utf-8') as f_err:
        
        for line in f_in:
            try:
                sample = json.loads(line)
                lang = sample['lang']
                code = sample.get('code_cleaned', '')
                prompt = sample.get('input', '')
                
                sample['kappa_cyclomatic'] = get_cyclomatic_complexity(code, lang)
                sample['coupling_depth'] = get_dependency_depth(code, lang)
                sample['m_mem_jaccard'] = get_jaccard_similarity(prompt, code)
                
                raw_e = sample.get('e_metric_refined', 0.0)
                lang_stats = stats.get(lang, {'mean': 0, 'std': 1})
                sample['e_norm'] = (raw_e - lang_stats['mean']) / lang_stats['std']
                
                f_out.write(json.dumps(sample) + '\n')
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"Progress: {processed_count} / 35,499...")
                        
            except Exception as e:
                f_err.write(f"Error processing sample: {e}\n")
                continue

    print(f"Feature Extraction Complete. {processed_count} records ready.")

if __name__ == "__main__":
    main()
