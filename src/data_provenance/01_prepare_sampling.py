"""
Step 1: Download stratified sample from OpenCodeInstruct.

Source: nvidia/OpenCodeInstruct on HuggingFace (~5M records, 50 parquet shards)
Output: stratified_sample_100k.jsonl (actual yield: 71,230 records)

KNOWN ISSUES (see data_provenance/README.md):
  - Language detection is keyword-based with high overlap risk
  - Python is NOT in the detection chain (lines 50-57 only match Java/C++/JS/Go)
  - ThreadPoolExecutor + as_completed = non-deterministic shard order
  - The 20k/language cap interacts with shard order to produce different samples

Original location: D:/ProgD/InstructionEntropy_Economics/prepare_sampling.py
Copied here for complete data provenance.
"""
import os
import zlib
import json
import requests
import io
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_url
import concurrent.futures

# CONFIG
DATASET_NAME = "nvidia/OpenCodeInstruct"
STRATIFIED_PULL_SIZE = 100000
LANGUAGES = ['python', 'java', 'javascript', 'cpp', 'go']
OUTPUT_PATH = "data/stratified_sample_100k.jsonl"

def get_complexity_e(instruction, response):
    if not instruction or not response:
        return 0
    c_instr = len(zlib.compress(instruction.encode('utf-8')))
    c_resp = len(zlib.compress(response.encode('utf-8')))
    return c_resp / c_instr if c_instr > 0 else 0

def process_shard(shard_idx):
    file_path = f"data/train-{shard_idx:05d}-of-00050.parquet"
    url = hf_hub_url(repo_id=DATASET_NAME, filename=file_path, repo_type="dataset")
    
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return []
        
        table = pq.read_table(io.BytesIO(resp.content))
        df = table.to_pandas()
        
        shard_samples = []
        for _, entry in df.iterrows():
            input_text = str(entry.get('input', ''))
            output_text = str(entry.get('output', ''))
            unit_tests = str(entry.get('unit_tests', ''))
            status_raw = str(entry.get('tests_execution_status', ''))

            if not unit_tests:
                continue

            found_lang = None
            search_text = (output_text + " " + input_text).lower()
            
            # KNOWN ISSUE: keyword-based detection with high overlap risk
            if any(x in search_text for x in ['java', 'public class', 'System.out', 'args[]', 'println']):
                found_lang = 'java'
            elif any(x in search_text for x in ['cpp', 'c++', 'iostream', '#include', 'std::', 'vector<']):
                found_lang = 'cpp'
            elif any(x in search_text for x in ['javascript', ' js ', 'const ', 'let ', 'function', '=>']):
                found_lang = 'javascript'
            elif any(x in search_text for x in ['go ', 'golang', 'package main', 'func ']):
                found_lang = 'go'
            
            # KNOWN ISSUE: Python is never matched here
            if not found_lang:
                continue
                
            shard_samples.append({
                'id': entry.get('id'),
                'lang': found_lang,
                'input': input_text,
                'output': output_text,
                'unit_tests': unit_tests,
                'e_metric': get_complexity_e(input_text, output_text)
            })
            
        return shard_samples
    except Exception as e:
        return []

def main():
    print(f"Downloading stratified sample from OpenCodeInstruct...")
    
    all_samples = []
    existing_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                s = json.loads(line)
                all_samples.append(s)
                existing_ids.add(s['id'])
    
    counts = {lang: 0 for lang in LANGUAGES}
    for s in all_samples:
        counts[s['lang']] += 1
    
    target_per_lang = 20000
    
    # KNOWN ISSUE: non-deterministic shard order
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_shard = {executor.submit(process_shard, idx): idx for idx in range(50)}
        
        for future in concurrent.futures.as_completed(future_to_shard):
            shard_idx = future_to_shard[future]
            shard_samples = future.result()
            
            new_added = 0
            for s in shard_samples:
                if s['id'] in existing_ids:
                    continue
                if counts[s['lang']] < target_per_lang:
                    all_samples.append(s)
                    existing_ids.add(s['id'])
                    counts[s['lang']] += 1
                    new_added += 1
            
            print(f"Shard {shard_idx} done. Added {new_added}. Counts: {counts}")
            
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                for s in all_samples:
                    f.write(json.dumps(s) + '\n')
            
            if all(counts[l] >= target_per_lang for l in ['java', 'javascript', 'cpp', 'go']):
                break

if __name__ == "__main__":
    main()
