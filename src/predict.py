import os
import sys
import time
import pickle
import csv
import sqlite3
import argparse
import gc
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

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

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
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

def run_test_inference(max_s1_limit=None, benchmark_mode=False, custom_threshold=None, output_matching_file=None):
    print("==================================================", flush=True)
    mode_str = f"BENCHMARK MODE (Limit {max_s1_limit:,})" if benchmark_mode else "FULL PRODUCTION TEST INFERENCE PIPELINE (DEADLINE MODE)"
    print(f"PHASE 4 — {mode_str}", flush=True)
    print("==================================================", flush=True)
    start_all = time.time()

    # 1. Load Trained Model & Evaluation Artifacts
    model_path = os.path.join(MODEL_DIR, "matcher_model.pkl")
    artifacts_path = os.path.join(MODEL_DIR, "eval_artifacts.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        print("Error: Trained model or evaluation artifacts missing. Run train_matcher.py first.")
        sys.exit(1)

    matcher = EntityMatcher.load(model_path)
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    optimal_threshold = custom_threshold if custom_threshold is not None else artifacts['best_thresh']
    print(f"Loaded champion matcher model ({matcher.model_type}) with decision threshold = {optimal_threshold:.3f}", flush=True)

    # 2. Step A: Scan test_source1.tsv to build Targeted Query Vocabulary
    print("Scanning test_source1.tsv to extract targeted query vocabulary...", flush=True)
    start_s1_scan = time.time()
    target_core_names = set()
    target_pins = set()
    target_first_nums = set()
    
    test_s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    
    with open(test_s1_path, "r", encoding="utf-8") as f_s1:
        reader = csv.reader(f_s1, delimiter="\t")
        header = next(reader, None)
        count = 0
        for row in reader:
            if len(row) < 4:
                continue
            s1_id = row[0].strip()
            bname = row[1]
            baddr = row[2]
            country = row[3]

            rec = precompute_record(s1_id, country, bname, baddr)
            if rec['core_name']:
                target_core_names.add(f"{rec['country']}_{rec['core_name']}")
            if rec['postal_clean']:
                target_pins.add(f"{rec['country']}_{rec['postal_clean']}")
            if rec['first_word']:
                for num in rec['addr_nums_set']:
                    target_first_nums.add(f"{rec['country']}_{rec['first_word']}_{num}")

            count += 1
            if max_s1_limit and count >= max_s1_limit:
                break

    print(f"Target Vocabulary extracted in {time.time() - start_s1_scan:.2f}s ({len(target_core_names):,} core names, {len(target_pins):,} pins, {len(target_first_nums):,} first_word-num keys).", flush=True)

    # 3. Setup SQLite DB for Zero-RAM Raw Attribute Storage
    db_exists_and_valid = False
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM records")
            cnt = cur.fetchone()[0]
            if cnt >= 9969000:
                db_exists_and_valid = True
                print(f"Found existing pre-indexed SQLite DB ({cnt:,} records). Skipping attribute indexing!", flush=True)
            else:
                conn.close()
                os.remove(DB_PATH)
        except Exception:
            if 'conn' in locals():
                conn.close()
            try:
                os.remove(DB_PATH)
            except Exception:
                pass

    if not db_exists_and_valid:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = OFF;")
        cur.execute("CREATE TABLE records (rec_id TEXT PRIMARY KEY, country TEXT, name TEXT, addr TEXT);")

    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS batch_ids (rec_id TEXT PRIMARY KEY);")

    # Integer ID mappings
    id_to_str = []
    str_to_id = {}

    opt_core_name_idx = defaultdict(list)
    opt_pin_idx = defaultdict(list)
    opt_first_num_idx = defaultdict(list)

    print("Building high-precision integer-mapped index from Test Source 2 and Source 3...", flush=True)
    start_idx = time.time()

    def index_and_cache_file(filename):
        filepath = os.path.join(TEST_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            count = 0
            db_batch = []
            
            for row in reader:
                if len(row) < 4:
                    continue
                rec_id = row[0].strip()
                name = row[1]
                addr = row[2]
                country = row[3].strip().upper() if len(row) > 3 else ""

                if rec_id not in str_to_id:
                    int_id = len(id_to_str)
                    str_to_id[rec_id] = int_id
                    id_to_str.append(rec_id)
                else:
                    int_id = str_to_id[rec_id]

                if not db_exists_and_valid:
                    db_batch.append((rec_id, country, name, addr))
                    if len(db_batch) >= 100000:
                        cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?);", db_batch)
                        db_batch.clear()

                name_clean = normalize_text(name)
                _, core_name, _ = extract_legal_suffixes(name)
                if core_name:
                    core_key = f"{country}_{core_name}"
                    if core_key in target_core_names:
                        opt_core_name_idx[core_key].append(int_id)

                pin = extract_pin(addr)
                if pin:
                    pin_key = f"{country}_{pin}"
                    if pin_key in target_pins:
                        opt_pin_idx[pin_key].append(int_id)

                first_word = name_clean.split()[0] if name_clean.split() else ""
                if first_word:
                    nums = extract_addr_numerics(addr)
                    for num in nums:
                        fn_key = f"{country}_{first_word}_{num}"
                        if fn_key in target_first_nums:
                            opt_first_num_idx[fn_key].append(int_id)

                count += 1
                if count % 2000000 == 0:
                    print(f"  [{filename}] Indexed {count:,} records...", flush=True)

            if not db_exists_and_valid and db_batch:
                cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?);", db_batch)
                db_batch.clear()

            if not db_exists_and_valid:
                conn.commit()

    index_and_cache_file("test_source2.tsv")
    index_and_cache_file("test_source3.tsv")

    if not db_exists_and_valid:
        print("Building SQLite index on rec_id...", flush=True)
        cur.execute("CREATE UNIQUE INDEX idx_rec_id ON records(rec_id);")
        conn.commit()

    print(f"High-Precision Targeted Integer Index built in {time.time() - start_idx:.2f}s ({len(id_to_str):,} total records mapped, RAM: {get_peak_memory_mb():.1f} MB).", flush=True)

    def fetch_precomputed_records_batch(needed_ids_str):
        precomputed = {}
        if not needed_ids_str:
            return precomputed
            
        cur.execute("DELETE FROM batch_ids;")
        cur.executemany("INSERT OR IGNORE INTO batch_ids VALUES (?);", [(cid,) for cid in needed_ids_str])
        cur.execute("SELECT r.rec_id, r.country, r.name, r.addr FROM records r JOIN batch_ids b ON r.rec_id = b.rec_id;")
        
        for r in cur.fetchall():
            rec_id, country, name, addr = r[0], r[1], r[2], r[3]
            precomputed[rec_id] = precompute_record(rec_id, country, name, addr)
            
        return precomputed

    # 4. Stream Process Test Source 1 & Write Submission Outputs
    print("Generating candidate pairs and match predictions for test_source1.tsv...", flush=True)
    start_infer = time.time()
    
    total_s1_processed = 0
    total_candidates_generated = 0
    total_matches_predicted = 0
    s1_with_matches = 0
    s1_no_matches = 0
    s2_matches = 0
    s3_matches = 0

    cand_counts_list = []

    if output_matching_file:
        matching_file = output_matching_file
        candidate_file = os.path.join(OUTPUT_DIR, "v3_dummy_candidate_pairs.tsv")
    elif benchmark_mode:
        matching_file = os.path.join(OUTPUT_DIR, "benchmark_matching_results.tsv")
        candidate_file = os.path.join(OUTPUT_DIR, "benchmark_candidate_pairs.tsv")
    else:
        matching_file = MATCHING_OUTPUT
        candidate_file = CANDIDATE_OUTPUT

    batch_size = 10000

    with open(matching_file, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_file, "w", encoding="utf-8", newline="") as f_cand:

        match_writer = csv.writer(f_match, delimiter="\t")
        cand_writer = csv.writer(f_cand, delimiter="\t")

        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        with open(test_s1_path, "r", encoding="utf-8") as f_s1:
            reader = csv.reader(f_s1, delimiter="\t")
            header = next(reader, None)

            batch_s1_recs = []
            
            for row in reader:
                if len(row) < 4:
                    continue
                s1_id = row[0].strip()
                bname = row[1]
                baddr = row[2]
                country = row[3]

                rec = precompute_record(s1_id, country, bname, baddr)
                batch_s1_recs.append((s1_id, rec))

                if len(batch_s1_recs) >= batch_size:
                    proc_match, proc_cand, c_gen, m_pred, w_match, no_match, s2_cnt, s3_cnt, batch_c_counts = process_s1_batch_deadline(
                        batch_s1_recs, fetch_precomputed_records_batch, opt_core_name_idx, opt_pin_idx, opt_first_num_idx, id_to_str, matcher, optimal_threshold
                    )
                    for r in proc_match:
                        match_writer.writerow(r)
                    for r in proc_cand:
                        cand_writer.writerow(r)

                    total_s1_processed += len(batch_s1_recs)
                    total_candidates_generated += c_gen
                    total_matches_predicted += m_pred
                    s1_with_matches += w_match
                    s1_no_matches += no_match
                    s2_matches += s2_cnt
                    s3_matches += s3_cnt
                    cand_counts_list.extend(batch_c_counts)

                    gc.collect()

                    elapsed_so_far = time.time() - start_infer
                    pct = (total_s1_processed / 1732544.0) * 100.0 if not max_s1_limit else (total_s1_processed / max_s1_limit) * 100.0
                    rate = total_s1_processed / elapsed_so_far
                    eta_sec = (1732544 - total_s1_processed) / rate if rate > 0 else 0
                    eta_hrs = eta_sec / 3600.0

                    print(f"  Processed {total_s1_processed:,} S1 [{pct:.1f}%] | Matches: {total_matches_predicted:,} | Rate: {rate:.1f} S1/s | ETA: {eta_hrs:.2f}h | RAM: {get_peak_memory_mb():.1f} MB", flush=True)
                    batch_s1_recs.clear()

                    if max_s1_limit and total_s1_processed >= max_s1_limit:
                        break

            if batch_s1_recs and (not max_s1_limit or total_s1_processed < max_s1_limit):
                proc_match, proc_cand, c_gen, m_pred, w_match, no_match, s2_cnt, s3_cnt, batch_c_counts = process_s1_batch_deadline(
                    batch_s1_recs, fetch_precomputed_records_batch, opt_core_name_idx, opt_pin_idx, opt_first_num_idx, id_to_str, matcher, optimal_threshold
                )
                for r in proc_match:
                    match_writer.writerow(r)
                for r in proc_cand:
                    cand_writer.writerow(r)

                total_s1_processed += len(batch_s1_recs)
                total_candidates_generated += c_gen
                total_matches_predicted += m_pred
                s1_with_matches += w_match
                s1_no_matches += no_match
                s2_matches += s2_cnt
                s3_matches += s3_cnt
                cand_counts_list.extend(batch_c_counts)
                batch_s1_recs.clear()

    if not benchmark_mode:
        conn.close()

    infer_runtime = time.time() - start_infer
    peak_mem = get_peak_memory_mb()

    mean_cands = float(np.mean(cand_counts_list)) if cand_counts_list else 0.0
    median_cands = float(np.median(cand_counts_list)) if cand_counts_list else 0.0
    p95_cands = float(np.percentile(cand_counts_list, 95)) if cand_counts_list else 0.0
    max_cands = int(np.max(cand_counts_list)) if cand_counts_list else 0

    print("\n" + "="*80, flush=True)
    print("PRODUCTION TEST INFERENCE SUMMARY & STATISTICAL METRICS", flush=True)
    print("="*80, flush=True)
    print(f"Total Test S1 Entities         : {total_s1_processed:,}")
    print(f"S1 Entities with >=1 Match     : {s1_with_matches:,}")
    print(f"S1 Entities with 0 Match       : {s1_no_matches:,}")
    print(f"Total Candidate Pairs Gen      : {total_candidates_generated:,}")
    print(f"Total Predicted Matches        : {total_matches_predicted:,}")
    print(f"  - Source 2 (S2) Matches      : {s2_matches:,}")
    print(f"  - Source 3 (S3) Matches      : {s3_matches:,}")
    print(f"Average Candidates per S1      : {mean_cands:.2f}")
    print(f"Median Candidates per S1       : {median_cands:.1f}")
    print(f"P95 Candidates per S1          : {p95_cands:.1f}")
    print(f"Maximum Candidates per S1      : {max_cands:,}")
    print(f"Total Inference Runtime        : {infer_runtime:.2f} s ({infer_runtime/3600.0:.2f} h)")
    print(f"Peak Memory Utilization        : {peak_mem:.1f} MB")
    print("="*80, flush=True)

    if not benchmark_mode:
        print("\nExecuting Post-Inference Integrity Verification Checks (A-I)...", flush=True)
        verify_integrity(MATCHING_OUTPUT, CANDIDATE_OUTPUT, set(str_to_id.keys()), total_s1_processed)

    return {
        'total_s1': total_s1_processed,
        'c_gen': total_candidates_generated,
        'm_pred': total_matches_predicted,
        'mean_cands': mean_cands,
        'total_time': infer_runtime,
        'peak_mem': peak_mem,
    }

def process_s1_batch_deadline(batch_s1_recs, fetch_precomputed_records_batch, opt_core_name_idx, opt_pin_idx, opt_first_num_idx, id_to_str, matcher, optimal_threshold):
    match_rows = []
    cand_rows = []
    c_gen_count = 0
    m_pred_count = 0
    w_match = 0
    no_match = 0
    s2_cnt = 0
    s3_cnt = 0
    batch_c_counts = []

    s1_dict_batch = {}
    pair_reasons_batch = {}
    needed_cand_ids_str = set()
    candidate_pairs_batch = []

    for s1_id, s1_rec in batch_s1_recs:
        s1_dict_batch[s1_id] = s1_rec
        cands_set_int = set()
        reasons_dict = defaultdict(set)

        if s1_rec['core_name']:
            core_key = f"{s1_rec['country']}_{s1_rec['core_name']}"
            for cid in opt_core_name_idx.get(core_key, []):
                cands_set_int.add(cid)
                reasons_dict[cid].add("EXACT_CORE_NAME")

        if s1_rec['postal_clean']:
            pin_key = f"{s1_rec['country']}_{s1_rec['postal_clean']}"
            for cid in opt_pin_idx.get(pin_key, []):
                cands_set_int.add(cid)
                reasons_dict[cid].add("POSTAL_CODE")

        if s1_rec['first_word']:
            for num in s1_rec['addr_nums_set']:
                fn_key = f"{s1_rec['country']}_{s1_rec['first_word']}_{num}"
                for cid in opt_first_num_idx.get(fn_key, []):
                    cands_set_int.add(cid)
                    reasons_dict[cid].add("FREQ_TOKEN_NUMERIC_COMPOSITE")

        c_list_int = sorted(list(cands_set_int))
        c_list_str = [id_to_str[i] for i in c_list_int]

        c_gen_count += len(c_list_int)
        batch_c_counts.append(len(c_list_int))

        cand_str = ",".join(c_list_str)
        cand_rows.append([s1_id, cand_str])

        for cid_int, cid_str in zip(c_list_int, c_list_str):
            needed_cand_ids_str.add(cid_str)
            candidate_pairs_batch.append((s1_id, cid_str))
            pair_reasons_batch[(s1_id, cid_str)] = reasons_dict[cid_int]

    if needed_cand_ids_str:
        cand_dict_batch = fetch_precomputed_records_batch(needed_cand_ids_str)
        
        # Fast Filter
        ambiguous = []
        for s1_id, cid_str in candidate_pairs_batch:
            s1_rec = s1_dict_batch[s1_id]
            cand_rec = cand_dict_batch.get(cid_str)
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

            ambiguous.append((s1_id, cid_str))

        if ambiguous:
            probs_batch = np.zeros(len(ambiguous), dtype=np.float32)
            chunk_size = 100000
            
            for i in range(0, len(ambiguous), chunk_size):
                pairs_sub = ambiguous[i:i+chunk_size]
                X_sub = extract_features_for_pairs(pairs_sub, s1_dict_batch, cand_dict_batch, pair_reasons_batch)
                probs_batch[i:i+len(pairs_sub)] = matcher.predict_proba(X_sub)
            
            s1_matched_map = defaultdict(list)
            for idx, (s1_id, cid_str) in enumerate(ambiguous):
                if probs_batch[idx] >= optimal_threshold:
                    s1_matched_map[s1_id].append(cid_str)
                    m_pred_count += 1
                    if cid_str.startswith("S2-"):
                        s2_cnt += 1
                    else:
                        s3_cnt += 1

            for s1_id, _ in batch_s1_recs:
                m_list = s1_matched_map.get(s1_id, [])
                if m_list:
                    w_match += 1
                else:
                    no_match += 1
                match_str = ",".join(m_list)
                match_rows.append([s1_id, match_str])
        else:
            for s1_id, _ in batch_s1_recs:
                no_match += 1
                match_rows.append([s1_id, ""])
    else:
        for s1_id, _ in batch_s1_recs:
            no_match += 1
            match_rows.append([s1_id, ""])

    return match_rows, cand_rows, c_gen_count, m_pred_count, w_match, no_match, s2_cnt, s3_cnt, batch_c_counts

def verify_integrity(matching_path, candidate_path, valid_s2_s3_ids, expected_s1_count):
    errors = []
    
    with open(matching_path, "r", encoding="utf-8") as f:
        h = f.readline().strip().split("\t")
        if h != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"matching_results.tsv header invalid: {h}")

    with open(candidate_path, "r", encoding="utf-8") as f:
        h = f.readline().strip().split("\t")
        if h != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"candidate_pairs.tsv header invalid: {h}")

    match_s1_set = set()
    match_dict = {}
    match_rows_count = 0
    with open(matching_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            match_rows_count += 1
            if not row:
                continue
            s1_id = row[0].strip()
            if s1_id in match_s1_set:
                errors.append(f"Duplicate S1 ID in matching_results.tsv: {s1_id}")
            match_s1_set.add(s1_id)
            m_list = [m.strip() for m in row[1].split(",") if m.strip()] if len(row) > 1 else []
            if len(m_list) != len(set(m_list)):
                errors.append(f"Duplicate matched ID inside row for S1 ID: {s1_id}")
            match_dict[s1_id] = set(m_list)

    cand_s1_set = set()
    cand_dict = {}
    cand_rows_count = 0
    with open(candidate_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            cand_rows_count += 1
            if not row:
                continue
            s1_id = row[0].strip()
            if s1_id in cand_s1_set:
                errors.append(f"Duplicate S1 ID in candidate_pairs.tsv: {s1_id}")
            cand_s1_set.add(s1_id)
            c_list = [c.strip() for c in row[1].split(",") if c.strip()] if len(row) > 1 else []
            if len(c_list) != len(set(c_list)):
                errors.append(f"Duplicate candidate ID inside row for S1 ID: {s1_id}")
            cand_dict[s1_id] = set(c_list)

    if match_rows_count != expected_s1_count:
        errors.append(f"matching_results.tsv row count ({match_rows_count:,}) != expected ({expected_s1_count:,})")

    if cand_rows_count != expected_s1_count:
        errors.append(f"candidate_pairs.tsv row count ({cand_rows_count:,}) != expected ({expected_s1_count:,})")

    for s1_id, m_set in match_dict.items():
        c_set = cand_dict.get(s1_id, set())
        diff = m_set - c_set
        if diff:
            errors.append(f"S1 {s1_id} has matched IDs not in candidate set: {diff}")
        for mid in m_set:
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                errors.append(f"S1 {s1_id} has invalid target prefix: {mid}")
            if valid_s2_s3_ids and mid not in valid_s2_s3_ids:
                errors.append(f"S1 {s1_id} matched ID {mid} not present in test S2/S3 files")

    if errors:
        print(f"FAIL — {len(errors)} integrity error(s) found during post-inference check:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)

    print("PASS — All post-inference integrity checks (A-I) passed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Production Test Inference Pipeline")
    parser.add_argument("--benchmark", type=int, default=None, help="Run benchmark on first N S1 entities")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (e.g. 0.85)")
    parser.add_argument("--output-matching", type=str, default=None, help="Custom matching output TSV file path")
    args = parser.parse_args()

    if args.benchmark:
        run_test_inference(max_s1_limit=args.benchmark, benchmark_mode=True, custom_threshold=args.threshold, output_matching_file=args.output_matching)
    else:
        run_test_inference(custom_threshold=args.threshold, output_matching_file=args.output_matching)
