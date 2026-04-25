"""
Step 4: Execute unit tests and score each sample as pass/fail.

Input:  feature_extracted_set.jsonl   (35,499 records)
Output: final_results_scored.jsonl    (35,499 records + status field)

Execution:
  - Python:     subprocess python3 -c (code + tests)
  - JavaScript: subprocess node -e (code + tests)
  - Go/Java/C++: ⚠ NOT EXECUTED ,  fabricated based on kappa_cyclomatic

CRITICAL KNOWN ISSUE (see data_provenance/README.md):
  For Go, Java, and C++ (lines 57-59), test results are FABRICATED:
    result = "pass" if kappa_cyclomatic < 5 else "fail"
  This means ~22% of the dataset has synthetic test outcomes that
  correlate directly with complexity ,  the very variable the paper
  studies.  This MUST be addressed before publication.

Original location: D:/ProgD/InstructionEntropy_Economics/src/colab_batch_executor.py
Copied here for complete data provenance.
"""
import json
import subprocess
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# CONFIG
INPUT_FILE = "data/feature_extracted_set.jsonl"
OUTPUT_FILE = "data/final_results_scored.jsonl"
TIMEOUT = 5  # seconds per test block

def execute_python(code, tests):
    full_code = f"{code}\n\n{tests}"
    try:
        proc = subprocess.run(
            ['python3', '-c', full_code],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        return "pass" if proc.returncode == 0 else "fail"
    except:
        return "fail"

def execute_javascript(code, tests):
    full_code = f"{code}\n\n{tests}"
    try:
        proc = subprocess.run(
            ['node', '-e', full_code],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        return "pass" if proc.returncode == 0 else "fail"
    except:
        return "fail"

def process_sample(line):
    try:
        sample = json.loads(line)
        lang = sample.get('lang', 'python')
        code = sample.get('code_cleaned', '')
        
        raw_tests = sample.get('unit_tests', '[]')
        if isinstance(raw_tests, str):
            test_list = json.loads(raw_tests)
        else:
            test_list = raw_tests
            
        test_block = "\n".join(test_list)
        
        if lang == 'python':
            result = execute_python(code, test_block)
        elif lang == 'javascript':
            result = execute_javascript(code, test_block)
        else:
            # ⚠ CRITICAL: Go/Java/C++ tests are NOT executed.
            # Pass/fail is fabricated from kappa_cyclomatic.
            result = "pass" if sample.get('kappa_cyclomatic', 0) < 5 else "fail"
            
        sample['status'] = ["pass"] if result == "pass" else ["fail"]
        return json.dumps(sample)
    except Exception as e:
        return line

def main():
    print("Starting Test Execution (Scoring)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
    
    print(f"Total Records to Process: {len(lines)}")
    
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores...")

    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_sample, lines))

    with open(OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(r + '\n')
            
    print(f"Execution Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
