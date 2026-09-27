import os
import sys
import time
import pandas as pd

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ANALYSIS_DIR)
sys.path.insert(0, BASE_DIR)

from src.normalization import clean_str
from src.blocking import BlockingIndexer, DEFAULT_CONFIG

TEST_S1 = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2 = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3 = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

def generate_test_candidates(max_rows=None):
    print(f"Generating Candidate Pairs for Test Set{' [SAMPLE MODE: ' + str(max_rows) + ' rows]' if max_rows else ' [FULL DATASET MODE]'}...")
    start_time = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Initialize Indexer & Build Inverted Indexes on Test S2 & S3
    indexer = BlockingIndexer(config=DEFAULT_CONFIG)
    indexer.build_index(TEST_S2, TEST_S3, max_rows=max_rows)
    
    # 2. Process Test S1 in Chunks & Write Candidate Pairs TSV
    print(f"Processing {os.path.basename(TEST_S1)} and writing {CANDIDATE_TSV}...")
    
    total_test_s1 = 0
    empty_cand_s1 = 0
    total_cands_written = 0
    
    with open(CANDIDATE_TSV, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        rows_processed = 0
        for chunk in pd.read_csv(TEST_S1, sep="\t", chunksize=100000, dtype=str, keep_default_na=False):
            if max_rows and rows_processed >= max_rows:
                break
            if max_rows and (rows_processed + len(chunk) > max_rows):
                chunk = chunk.iloc[:max_rows - rows_processed]
                
            rows_processed += len(chunk)
            total_test_s1 += len(chunk)
            
            lines_to_write = []
            for row in chunk.itertuples(index=False):
                s1_id = row.entity_id
                bname = row.business_name
                baddr = row.business_address
                country = row.country
                
                # Final combined blocking strategy
                cands_dict = indexer.get_candidate_matches_for_s1(
                    bname, baddr, country,
                    strategies=["EXACT_CORE_NAME", "NAME_TOKEN", "ADDRESS_TOKEN", "POSTAL_CODE"]
                )
                
                cand_ids = sorted(list(cands_dict.keys()))
                num_cands = len(cand_ids)
                total_cands_written += num_cands
                
                if num_cands == 0:
                    empty_cand_s1 += 1
                    cand_str = ""
                else:
                    cand_str = ",".join(cand_ids)
                    
                lines_to_write.append(f"{s1_id}\t{cand_str}\n")
                
            f_out.writelines(lines_to_write)
            
    elapsed = time.time() - start_time
    avg_cands = (total_cands_written / total_test_s1) if total_test_s1 else 0.0
    
    print(f"Test Candidate Generation Complete in {elapsed:.2f}s!")
    print(f"Total Test S1 Entities: {total_test_s1:,}")
    print(f"S1 Entities with 0 Candidates: {empty_cand_s1:,} ({empty_cand_s1/total_test_s1*100:.2f}%)")
    print(f"Total Candidate Pairs Generated: {total_cands_written:,}")
    print(f"Average Candidates per S1: {avg_cands:.2f}")
    print(f"Output saved to: {CANDIDATE_TSV} (size: {os.path.getsize(CANDIDATE_TSV):,} bytes)")

if __name__ == "__main__":
    max_rows = None
    if "--sample" in sys.argv:
        max_rows = 5000
    for idx, arg in enumerate(sys.argv):
        if arg == "--max-rows" and idx + 1 < len(sys.argv):
            max_rows = int(sys.argv[idx + 1])
            
    generate_test_candidates(max_rows=max_rows)
