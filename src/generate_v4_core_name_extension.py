import os
import sys
import time
import csv
import sqlite3
from collections import defaultdict

csv.field_size_limit(100 * 1024 * 1024)

sys.path.insert(0, os.path.abspath("."))
from src.normalization import extract_legal_suffixes

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

V1_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_INPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
V4_CANDIDATE_PATH = os.path.join(OUTPUT_DIR, "matching_results_v4_candidate.tsv")

TEST_S1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

def norm_country(c):
    c = str(c).strip().upper()
    if c in ["FRANCE", "FR"]: return "FR"
    if c in ["INDIA", "IN"]: return "IN"
    if c in ["US", "USA", "UNITED STATES"]: return "US"
    return c

def get_core_name(name):
    _, core_name, _ = extract_legal_suffixes(name)
    return core_name

def generate_v4_candidate():
    print("==================================================", flush=True)
    print("GENERATING DEFENSIBLE V4 CANDIDATE (EXACT CORE-NAME EXTENSION)", flush=True)
    print("==================================================", flush=True)
    t0 = time.time()

    # 1. Load V1 Predictions
    print(f"Loading V1 predictions from {V1_PATH}...", flush=True)
    v1_preds = {}
    v1_order = []
    v1_total_matches = 0
    with open(V1_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            m_ids = [m.strip() for m in row[1].split(",") if m.strip()] if len(row) > 1 else []
            v1_preds[s1_id] = m_ids
            v1_order.append(s1_id)
            v1_total_matches += len(m_ids)

    print(f"Loaded {len(v1_order):,} S1 entities ({v1_total_matches:,} V1 matches).", flush=True)

    # 2. Extract Core Name and Country for S1
    print("Extracting core names for S1 entities...", flush=True)
    s1_info = {}
    target_core_keys = set()
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 4: continue
            s1_id = row[0].strip()
            country = norm_country(row[3])
            core_name = get_core_name(row[1])
            s1_info[s1_id] = (country, core_name)
            if core_name:
                target_core_keys.add(f"{country}_{core_name}")

    print(f"Extracted attributes for {len(s1_info):,} S1 entities ({len(target_core_keys):,} core keys).", flush=True)

    # 3. Stream S2 & S3 to index exact core names
    print("Indexing S2 and S3 for target core names...", flush=True)
    t_idx = time.time()
    core_name_index = defaultdict(list)

    def index_target_file(filepath):
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                rec_id = row[0].strip()
                country = norm_country(row[3])
                core_name = get_core_name(row[1])
                if core_name:
                    key = f"{country}_{core_name}"
                    if key in target_core_keys:
                        core_name_index[key].append(rec_id)
                count += 1
                if count % 2000000 == 0:
                    print(f"  [{os.path.basename(filepath)}] {count:,} records processed...", flush=True)

    index_target_file(TEST_S2_PATH)
    index_target_file(TEST_S3_PATH)
    print(f"Index built in {time.time() - t_idx:.2f}s.", flush=True)

    # 4. Stream candidate_pairs.tsv and extend V1 predictions
    print(f"Streaming {CANDIDATE_INPUT} and extending V1 predictions...", flush=True)
    t_gen = time.time()
    v4_total_matches = 0
    added_matches_cnt = 0
    affected_s1_cnt = 0

    with open(CANDIDATE_INPUT, "r", encoding="utf-8") as f_cand, \
         open(V4_CANDIDATE_PATH, "w", encoding="utf-8", newline="") as f_out:

        r_cand = csv.reader(f_cand, delimiter="\t")
        next(r_cand, None) # skip header

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for row in r_cand:
            if not row: continue
            s1_id = row[0].strip()
            c_list = set(c.strip() for c in row[1].split(",") if c.strip()) if len(row) > 1 else set()

            v1_matches = v1_preds.get(s1_id, [])
            country, core_name = s1_info.get(s1_id, ("", ""))

            final_matches = list(v1_matches) # start with V1
            existing_set = set(v1_matches)

            if core_name and c_list:
                key = f"{country}_{core_name}"
                candidates_with_same_core = core_name_index.get(key, [])
                for cid in candidates_with_same_core:
                    if cid in c_list and cid not in existing_set:
                        final_matches.append(cid)
                        existing_set.add(cid)
                        added_matches_cnt += 1

            if len(final_matches) > len(v1_matches):
                affected_s1_cnt += 1

            v4_total_matches += len(final_matches)
            writer.writerow([s1_id, ",".join(final_matches)])

    total_time = time.time() - t0
    print("\n" + "="*80, flush=True)
    print("V4 CANDIDATE GENERATION COMPLETED SUCCESSFULLY", flush=True)
    print("="*80, flush=True)
    print(f"Output File           : {V4_CANDIDATE_PATH}")
    print(f"Total S1 Entities     : {len(v1_order):,}")
    print(f"V1 Matches            : {v1_total_matches:,}")
    print(f"V4 Matches            : {v4_total_matches:,}")
    print(f"Added Matches         : {added_matches_cnt:,}")
    print(f"Affected S1 Entities  : {affected_s1_cnt:,} ({affected_s1_cnt/len(v1_order)*100:.2f}%)")
    print(f"Total Execution Time  : {total_time:.2f} s ({total_time/60.0:.2f} min)")

if __name__ == "__main__":
    generate_v4_candidate()
