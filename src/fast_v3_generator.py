import os
import sys
import time
import pickle
import csv
csv.field_size_limit(100 * 1024 * 1024)
import sqlite3
import gc
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from src.normalization import (
    clean_str,
    normalize_text,
    extract_pin,
    extract_legal_suffixes,
    extract_addr_numerics,
    extract_location_tokens,
    tokenize_name,
    tokenize_address,
)
from src.features import FEATURE_NAMES, compute_pair_features, extract_features_for_pairs
from src.matcher import EntityMatcher

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MATCHING_V3_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results_v3.tsv")
CANDIDATE_INPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
DB_PATH = os.path.join(OUTPUT_DIR, "test_records_cache.db")

def get_peak_memory_mb():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def precompute_record(rec_id, country, name, address):
    country_clean = str(country).strip().upper() if country else ""
    name_clean = normalize_text(name)
    address_clean = normalize_text(address)
    _, core_name, _ = extract_legal_suffixes(name)
    pin = extract_pin(address)
    
    name_tokens = tokenize_name(name, min_len=4)
    addr_tokens = tokenize_address(address, min_len=4)
    addr_nums = extract_addr_numerics(address_clean)
    loc_tokens = extract_location_tokens(address_clean, country_clean)
    first_word = name_clean.split()[0] if name_clean.split() else ""
    
    source_prefix = "s2" if rec_id.startswith("S2-") else ("s3" if rec_id.startswith("S3-") else "s1")

    return {
        'id': rec_id,
        'country': country_clean,
        'name_clean': name_clean,
        'address_clean': address_clean,
        'core_name': core_name,
        'postal_clean': pin if pin else "",
        'first_word': first_word,
        'name_tokens_set': set(name_tokens),
        'addr_tokens_set': set(addr_tokens),
        'addr_nums_set': set(addr_nums),
        'location_tokens_set': set(loc_tokens),
        'source_prefix': source_prefix,
    }

def generate_v3_predictions(threshold=0.850):
    print("==================================================", flush=True)
    print(f"GENERATING V3 MATCHING PREDICTIONS (Threshold = {threshold:.3f})", flush=True)
    print("==================================================", flush=True)
    start_time = time.time()

    # 1. Load Model
    model_path = os.path.join(MODEL_DIR, "matcher_model.pkl")
    if not os.path.exists(model_path):
        print("Error: matcher_model.pkl not found.")
        sys.exit(1)
    
    matcher = EntityMatcher.load(model_path)
    print(f"Loaded trained LightGBM model ({matcher.model_type}).", flush=True)

    # 2. Check SQLite DB Cache for candidate attributes
    if not os.path.exists(DB_PATH):
        print(f"Error: SQLite record cache DB missing at {DB_PATH}.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS batch_ids (rec_id TEXT PRIMARY KEY);")

    def fetch_cand_records_batch(needed_ids_set):
        precomputed = {}
        if not needed_ids_set:
            return precomputed
        
        cur.execute("DELETE FROM batch_ids;")
        cur.executemany("INSERT OR IGNORE INTO batch_ids VALUES (?);", [(cid,) for cid in needed_ids_set])
        cur.execute("SELECT r.rec_id, r.country, r.name, r.addr FROM records r JOIN batch_ids b ON r.rec_id = b.rec_id;")
        
        for r in cur.fetchall():
            rec_id, country, name, addr = r[0], r[1], r[2], r[3]
            precomputed[rec_id] = precompute_record(rec_id, country, name, addr)
            
        return precomputed

    # 3. Dual-Stream test_source1.tsv and candidate_pairs.tsv in 20k batches with 100k pair sub-chunking
    test_s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    print(f"Dual-streaming S1 records and candidate pairs with 100k pair sub-chunking (RAM footprint < 150 MB)...", flush=True)
    
    total_s1_processed = 0
    total_predicted_matches = 0
    s1_with_matches = 0
    s1_no_matches = 0
    s2_matches = 0
    s3_matches = 0

    us_matches = 0
    india_matches = 0
    france_matches = 0
    other_matches = 0

    start_scoring = time.time()
    batch_size = 20000
    pair_chunk_size = 100000
    TOTAL_S1_EXPECTED = 1732544

    with open(test_s1_path, "r", encoding="utf-8") as f_s1, \
         open(CANDIDATE_INPUT, "r", encoding="utf-8") as f_cand, \
         open(MATCHING_V3_OUTPUT, "w", encoding="utf-8", newline="") as f_out:

        r_s1 = csv.reader(f_s1, delimiter="\t")
        r_cand = csv.reader(f_cand, delimiter="\t")

        next(r_s1, None)    # header
        next(r_cand, None)  # header

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        batch_s1_records = {}
        batch_cand_map = {}
        batch_s1_order = []

        while True:
            for _ in range(batch_size):
                row_s1 = next(r_s1, None)
                row_cand = next(r_cand, None)
                if not row_s1 or not row_cand:
                    break

                s1_id = row_s1[0].strip()
                s1_rec = precompute_record(s1_id, row_s1[3] if len(row_s1)>3 else "", row_s1[1], row_s1[2])
                c_list = [c.strip() for c in row_cand[1].split(",") if c.strip()] if len(row_cand) > 1 else []

                batch_s1_records[s1_id] = s1_rec
                batch_cand_map[s1_id] = c_list
                batch_s1_order.append(s1_id)

            if not batch_s1_order:
                break

            needed_cand_ids = set()
            for c_list in batch_cand_map.values():
                needed_cand_ids.update(c_list)

            cand_dict_batch = fetch_cand_records_batch(needed_cand_ids)

            batch_pairs = []
            pair_reasons_map = {}

            for s1_id in batch_s1_order:
                s1_rec = batch_s1_records[s1_id]
                c_list = batch_cand_map[s1_id]

                for cid in c_list:
                    cand_rec = cand_dict_batch.get(cid)
                    if not cand_rec:
                        continue

                    if s1_rec['country'] and cand_rec['country'] and s1_rec['country'] != cand_rec['country']:
                        continue

                    same_core_name = bool(s1_rec['core_name'] and s1_rec['core_name'] == cand_rec['core_name'])
                    same_postal = bool(s1_rec['postal_clean'] and s1_rec['postal_clean'] == cand_rec['postal_clean'])

                    s1_ntoks = s1_rec['name_tokens_set']
                    cand_ntoks = cand_rec['name_tokens_set']
                    common_name_toks = len(s1_ntoks.intersection(cand_ntoks))
                    union_name_toks = len(s1_ntoks.union(cand_ntoks))
                    jaccard_name = common_name_toks / float(union_name_toks) if union_name_toks > 0 else 0.0

                    s1_nums = s1_rec['addr_nums_set']
                    cand_nums = cand_rec['addr_nums_set']
                    common_nums = len(s1_nums.intersection(cand_nums))

                    if not same_core_name and not same_postal and common_nums == 0:
                        if jaccard_name < 0.20:
                            continue
                        if s1_rec['first_word'] and cand_rec['first_word'] and s1_rec['first_word'] != cand_rec['first_word'] and jaccard_name < 0.40:
                            continue

                    reasons = set()
                    if same_core_name:
                        reasons.add("EXACT_CORE_NAME")
                    if same_postal:
                        reasons.add("POSTAL_CODE")

                    batch_pairs.append((s1_id, cid))
                    pair_reasons_map[(s1_id, cid)] = reasons

            if batch_pairs:
                probs_batch = np.zeros(len(batch_pairs), dtype=np.float32)

                for p_idx in range(0, len(batch_pairs), pair_chunk_size):
                    sub_pairs = batch_pairs[p_idx : p_idx + pair_chunk_size]
                    X_sub = extract_features_for_pairs(sub_pairs, batch_s1_records, cand_dict_batch, pair_reasons_map)
                    probs_batch[p_idx : p_idx + len(sub_pairs)] = matcher.predict_proba(X_sub)

                s1_matches_map = defaultdict(list)
                for idx, (s1_id, cid) in enumerate(batch_pairs):
                    if probs_batch[idx] >= threshold:
                        s1_matches_map[s1_id].append(cid)

                for s1_id in batch_s1_order:
                    m_ids = s1_matches_map.get(s1_id, [])
                    if m_ids:
                        s1_with_matches += 1
                        total_predicted_matches += len(m_ids)
                        
                        s1_country = batch_s1_records[s1_id]['country']
                        if s1_country == "US":
                            us_matches += len(m_ids)
                        elif s1_country == "IN":
                            india_matches += len(m_ids)
                        elif s1_country == "FR":
                            france_matches += len(m_ids)
                        else:
                            other_matches += len(m_ids)

                        for mid in m_ids:
                            if mid.startswith("S2-"):
                                s2_matches += 1
                            else:
                                s3_matches += 1

                        match_str = ",".join(m_ids)
                        writer.writerow([s1_id, match_str])
                    else:
                        s1_no_matches += 1
                        writer.writerow([s1_id, ""])
            else:
                for s1_id in batch_s1_order:
                    s1_no_matches += 1
                    writer.writerow([s1_id, ""])

            total_s1_processed += len(batch_s1_order)
            batch_s1_records.clear()
            batch_cand_map.clear()
            batch_s1_order.clear()
            cand_dict_batch.clear()
            gc.collect()

            elapsed = time.time() - start_scoring
            rate = total_s1_processed / elapsed if elapsed > 0 else 0
            pct = (total_s1_processed / TOTAL_S1_EXPECTED) * 100.0
            eta_sec = (TOTAL_S1_EXPECTED - total_s1_processed) / rate if rate > 0 else 0
            eta_hrs = eta_sec / 3600.0

            print(f"  Processed {total_s1_processed:,}/{TOTAL_S1_EXPECTED:,} S1 [{pct:.1f}%] | Matches: {total_predicted_matches:,} | Rate: {rate:.1f} S1/s | ETA: {eta_hrs:.2f}h | RAM: {get_peak_memory_mb():.1f} MB", flush=True)

    conn.close()

    total_time = time.time() - start_time
    print("\n" + "="*80, flush=True)
    print("V3 PREDICTION GENERATION COMPLETED", flush=True)
    print("="*80, flush=True)
    print(f"Output File             : {MATCHING_V3_OUTPUT}")
    print(f"Total Test S1 Entities   : {total_s1_processed:,}")
    print(f"Matched S1 Count        : {s1_with_matches:,} ({s1_with_matches/total_s1_processed*100:.2f}%)")
    print(f"Empty/Unmatched S1 Count: {s1_no_matches:,} ({s1_no_matches/total_s1_processed*100:.2f}%)")
    print(f"Total Predicted Matches : {total_predicted_matches:,}")
    print(f"  - Source 2 (S2) Matches: {s2_matches:,}")
    print(f"  - Source 3 (S3) Matches: {s3_matches:,}")
    print(f"Average Matches per S1  : {total_predicted_matches/total_s1_processed:.3f}")
    print(f"Matches by Country      : US={us_matches:,}, India={india_matches:,}, France={france_matches:,}, Other={other_matches:,}")
    print(f"Total Runtime           : {total_time:.2f} s ({total_time/60.0:.2f} min)")

if __name__ == "__main__":
    generate_v3_predictions(threshold=0.850)
