import os
import sys
import time
import pickle
import random
import csv
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
from rapidfuzz.distance import JaroWinkler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def calc_f05(p, r):
    if (0.25 * p + r) == 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)

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

def run_pipeline():
    print("==================================================", flush=True)
    print("PHASE 3 — TRAINING & ML MATCHING PIPELINE", flush=True)
    print("==================================================", flush=True)
    start_all = time.time()

    # 1. Load Ground Truth
    print("Loading Ground Truth...", flush=True)
    s1_gt_map = defaultdict(set)
    gt_path = os.path.join(DATA_DIR, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                s1_id = row[0].strip()
                m_ids = [m.strip() for m in row[1].split(",") if m.strip()]
                s1_gt_map[s1_id].update(m_ids)

    # 2. Load 10,000 S1 records for training/validation sample
    print("Loading 10,000 S1 records for train/val pool...", flush=True)
    s1_dict = {}
    s1_ids = []
    
    target_tokens = set()
    target_core_names = set()
    target_pins = set()
    target_nums = set()

    s1_path = os.path.join(DATA_DIR, "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        count = 0
        for row in reader:
            if len(row) < 4:
                continue
            rec_id = row[0].strip()
            name = row[1]
            addr = row[2]
            country = row[3]
            
            rec = precompute_record(rec_id, country, name, addr)
            s1_dict[rec_id] = rec
            s1_ids.append(rec_id)
            
            target_tokens.update(rec['name_tokens_set'])
            if rec['core_name']:
                target_core_names.add(f"{rec['country']}_{rec['core_name']}")
            if rec['postal_clean']:
                target_pins.add(f"{rec['country']}_{rec['postal_clean']}")
            for num in rec['addr_nums_set']:
                target_nums.add(f"{rec['country']}_{num}")
                
            count += 1
            if count >= 10000:
                break

    print(f"Target S1 Vocabularies: {len(target_tokens):,} tokens, {len(target_core_names):,} core names, {len(target_pins):,} pins, {len(target_nums):,} numbers.", flush=True)

    # 3. Stream-Index Source 2 and Source 3 & Candidate Generation
    print("Building targeted inverted index from Source 2 and Source 3...", flush=True)
    start_cand_gen = time.time()
    
    name_token_idx = defaultdict(set)
    core_name_idx = defaultdict(set)
    pin_idx = defaultdict(set)
    addr_num_idx = defaultdict(set)
    token_df = Counter()
    
    raw_records = {} # rec_id -> (country, name, addr)
    cand_dict = {}   # rec_id -> precomputed record dict
    total_s2_records = 0
    total_s3_records = 0

    def index_file(filename):
        nonlocal total_s2_records, total_s3_records
        filepath = os.path.join(DATA_DIR, filename)
        is_s2 = "source2" in filename
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            count = 0
            for row in reader:
                if len(row) < 4:
                    continue
                rec_id = row[0].strip()
                name = row[1]
                addr = row[2]
                country = row[3].strip().upper() if len(row) > 3 else ""
                
                if is_s2:
                    total_s2_records += 1
                else:
                    total_s3_records += 1

                name_norm = normalize_text(name)
                ntokens = tokenize_name(name, min_len=4)
                for tok in ntokens:
                    token_df[tok] += 1
                    if tok in target_tokens:
                        name_token_idx[tok].add(rec_id)

                _, core_name, _ = extract_legal_suffixes(name)
                if core_name:
                    core_key = f"{country}_{core_name}"
                    if core_key in target_core_names:
                        core_name_idx[core_key].add(rec_id)

                pin = extract_pin(addr)
                if pin:
                    pin_key = f"{country}_{pin}"
                    if pin_key in target_pins:
                        pin_idx[pin_key].add(rec_id)

                nums = extract_addr_numerics(addr)
                for num in nums:
                    num_key = f"{country}_{num}"
                    if num_key in target_nums:
                        addr_num_idx[num_key].add(rec_id)

                count += 1
                if count % 2000000 == 0:
                    print(f"  [{filename}] {count:,} rows indexed...", flush=True)

    index_file("train_source2.tsv")
    index_file("train_source3.tsv")

    # Fast Candidate Generation Logic (Var5 Configuration)
    def get_var5_candidates(norm_s1, max_df=10000, max_df_intersect=50000):
        candidates = set()
        if norm_s1['core_name']:
            core_key = f"{norm_s1['country']}_{norm_s1['core_name']}"
            candidates.update(core_name_idx.get(core_key, set()))

        if norm_s1['postal_clean']:
            pin_key = f"{norm_s1['country']}_{norm_s1['postal_clean']}"
            candidates.update(pin_idx.get(pin_key, set()))

        s1_nums = norm_s1['addr_nums_set']
        for tok in norm_s1['name_tokens_set']:
            df = token_df[tok]
            if df <= max_df:
                candidates.update(name_token_idx.get(tok, set()))
            elif df <= max_df_intersect and s1_nums:
                freq_cands = name_token_idx.get(tok, set())
                for num in s1_nums:
                    num_key = f"{norm_s1['country']}_{num}"
                    num_cands = addr_num_idx.get(num_key, set())
                    candidates.update(freq_cands.intersection(num_cands))

        return candidates

    print("Generating candidate sets per S1 entity...", flush=True)
    s1_candidates_map = {}
    needed_cand_ids = set()
    total_candidates_generated = 0

    for s1_id in s1_ids:
        norm_s1 = s1_dict[s1_id]
        cands = get_var5_candidates(norm_s1)
        s1_candidates_map[s1_id] = cands
        needed_cand_ids.update(cands)
        needed_cand_ids.update(s1_gt_map.get(s1_id, set()))
        total_candidates_generated += len(cands)

    cand_gen_runtime = time.time() - start_cand_gen
    print(f"Candidate generation completed in {cand_gen_runtime:.2f}s ({total_candidates_generated:,} candidates generated for {len(s1_ids):,} S1 entities).", flush=True)

    print(f"Collected {len(needed_cand_ids):,} target candidate & GT record IDs. Loading attributes...", flush=True)

    def load_candidate_attributes(filename):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                rec_id = row[0].strip()
                if rec_id in needed_cand_ids:
                    raw_records[rec_id] = (row[3].strip().upper() if len(row) > 3 else "", row[1], row[2])

    load_candidate_attributes("train_source2.tsv")
    load_candidate_attributes("train_source3.tsv")
    print(f"Loaded attributes for {len(raw_records):,} required candidate records.", flush=True)

    def get_or_precompute_record(rec_id):
        if rec_id not in cand_dict:
            country, name, addr = raw_records.get(rec_id, ("", "", ""))
            cand_dict[rec_id] = precompute_record(rec_id, country, name, addr)
        return cand_dict[rec_id]

    def get_blocking_reasons(norm_s1, cand_id):
        reasons = set()
        if norm_s1['core_name']:
            core_key = f"{norm_s1['country']}_{norm_s1['core_name']}"
            if cand_id in core_name_idx.get(core_key, set()):
                reasons.add("EXACT_CORE_NAME")
        if norm_s1['postal_clean']:
            pin_key = f"{norm_s1['country']}_{norm_s1['postal_clean']}"
            if cand_id in pin_idx.get(pin_key, set()):
                reasons.add("POSTAL_CODE")
        for tok in norm_s1['name_tokens_set']:
            if token_df[tok] <= 10000 and cand_id in name_token_idx.get(tok, set()):
                reasons.add("NAME_TOKEN")
            elif token_df[tok] <= 50000 and norm_s1['addr_nums_set']:
                if cand_id in name_token_idx.get(tok, set()):
                    reasons.add("FREQ_TOKEN_NUMERIC_COMPOSITE")
        return reasons

    # 4. Entity-Level Train / Validation Split (80% Train / 20% Val)
    random.seed(42)
    np.random.seed(42)
    shuffled_s1 = list(s1_ids)
    random.shuffle(shuffled_s1)
    
    split_idx = int(0.8 * len(shuffled_s1))
    train_s1_set = set(shuffled_s1[:split_idx])
    val_s1_set = set(shuffled_s1[split_idx:])

    print(f"\nDataset Split: Train S1 Entities = {len(train_s1_set):,}, Val S1 Entities = {len(val_s1_set):,}", flush=True)

    # 5. Negative Sampling Strategy Experiments
    print("\n" + "="*80, flush=True)
    print("NEGATIVE SAMPLING STRATEGY ABLATION & EXPERIMENTS", flush=True)
    print("="*80, flush=True)

    neg_strategies = ["Uniform", "Hard", "Mixed"]
    neg_results = {}

    for strat in neg_strategies:
        pos_cnt = 0
        neg_cnt = 0
        hard_neg_cnt = 0
        
        for s1_id in train_s1_set:
            norm_s1 = s1_dict[s1_id]
            gt_targets = s1_gt_map.get(s1_id, set())
            cands = s1_candidates_map[s1_id]
            
            pos_cands = cands.intersection(gt_targets)
            neg_cands = list(cands - gt_targets)
            pos_cnt += len(pos_cands)

            if not neg_cands:
                continue

            if strat == "Uniform":
                sample_size = min(len(neg_cands), max(15 * len(pos_cands), 30))
                sampled = random.sample(neg_cands, sample_size)
                neg_cnt += len(sampled)
                hard_neg_cnt += 0
            elif strat == "Hard":
                hard_negs = []
                for cid in neg_cands:
                    crec = get_or_precompute_record(cid)
                    has_tok_inter = bool(norm_s1['name_tokens_set'].intersection(crec['name_tokens_set']))
                    num_overlap = bool(norm_s1['addr_nums_set'].intersection(crec['addr_nums_set']))
                    postal_match = bool(norm_s1['postal_clean'] and norm_s1['postal_clean'] == crec['postal_clean'])
                    if has_tok_inter or num_overlap or postal_match:
                        hard_negs.append(cid)
                sample_size = min(len(hard_negs), max(15 * len(pos_cands), 30)) if hard_negs else min(len(neg_cands), 30)
                sampled = random.sample(hard_negs, sample_size) if hard_negs else random.sample(neg_cands, sample_size)
                neg_cnt += len(sampled)
                hard_neg_cnt += len(sampled) if hard_negs else 0
            elif strat == "Mixed":
                hard_negs = []
                for cid in neg_cands:
                    crec = get_or_precompute_record(cid)
                    has_tok_inter = bool(norm_s1['name_tokens_set'].intersection(crec['name_tokens_set']))
                    num_overlap = bool(norm_s1['addr_nums_set'].intersection(crec['addr_nums_set']))
                    if has_tok_inter or num_overlap:
                        hard_negs.append(cid)
                half = max(10, (15 * len(pos_cands)) // 2)
                h_sample = random.sample(hard_negs, min(len(hard_negs), half)) if hard_negs else []
                u_sample = random.sample(neg_cands, min(len(neg_cands), half))
                sampled = list(set(h_sample + u_sample))
                neg_cnt += len(sampled)
                hard_neg_cnt += len(h_sample)

        pos_neg_ratio = pos_cnt / neg_cnt if neg_cnt > 0 else 0.0
        hard_prop = (hard_neg_cnt / neg_cnt * 100.0) if neg_cnt > 0 else 0.0
        neg_results[strat] = {
            'pos_count': pos_cnt,
            'neg_count': neg_cnt,
            'pos_neg_ratio': pos_neg_ratio,
            'hard_neg_prop': hard_prop,
        }

        print(f"Strategy {strat:<10} | Positives: {pos_cnt:6,} | Negatives: {neg_cnt:7,} | Pos/Neg Ratio: 1:{1/pos_neg_ratio:5.1f} | Hard Neg Proportion: {hard_prop:5.1f}%", flush=True)

    # 6. Construct Final Training & Validation Matrices (Mixed Negative Sampling)
    print("\nConstructing Final Train & Validation Feature Matrices...", flush=True)
    train_pairs, train_labels = [], []
    val_pairs, val_labels = [], []
    pair_reasons_map = {}
    
    train_pos_cnt = 0
    train_neg_cnt = 0
    val_pos_cnt = 0
    val_neg_cnt = 0

    for s1_id in s1_ids:
        norm_s1 = s1_dict[s1_id]
        gt_targets = s1_gt_map.get(s1_id, set())
        cands = s1_candidates_map[s1_id]
        is_train = s1_id in train_s1_set
        
        pos_cands = cands.intersection(gt_targets)
        neg_cands = list(cands - gt_targets)
        
        if is_train:
            for cand_id in pos_cands:
                get_or_precompute_record(cand_id)
                train_pairs.append((s1_id, cand_id))
                train_labels.append(1)
                train_pos_cnt += 1
                pair_reasons_map[(s1_id, cand_id)] = get_blocking_reasons(norm_s1, cand_id)
            
            if neg_cands:
                hard_negs = [cid for cid in neg_cands if (bool(norm_s1['name_tokens_set'].intersection(get_or_precompute_record(cid)['name_tokens_set'])) or bool(norm_s1['addr_nums_set'].intersection(get_or_precompute_record(cid)['addr_nums_set'])))]
                half = max(10, (15 * len(pos_cands)) // 2)
                h_sample = random.sample(hard_negs, min(len(hard_negs), half)) if hard_negs else []
                u_sample = random.sample(neg_cands, min(len(neg_cands), half))
                sampled_negs = list(set(h_sample + u_sample))

                for cand_id in sampled_negs:
                    get_or_precompute_record(cand_id)
                    train_pairs.append((s1_id, cand_id))
                    train_labels.append(0)
                    train_neg_cnt += 1
                    pair_reasons_map[(s1_id, cand_id)] = get_blocking_reasons(norm_s1, cand_id)
        else:
            for cand_id in pos_cands:
                get_or_precompute_record(cand_id)
                val_pairs.append((s1_id, cand_id))
                val_labels.append(1)
                val_pos_cnt += 1
                pair_reasons_map[(s1_id, cand_id)] = get_blocking_reasons(norm_s1, cand_id)
            
            if neg_cands:
                val_sampled_negs = random.sample(neg_cands, min(len(neg_cands), 150))
                for cand_id in val_sampled_negs:
                    get_or_precompute_record(cand_id)
                    val_pairs.append((s1_id, cand_id))
                    val_labels.append(0)
                    val_neg_cnt += 1
                    pair_reasons_map[(s1_id, cand_id)] = get_blocking_reasons(norm_s1, cand_id)

    print(f"Final Train Pair Count: {len(train_pairs):,} (Positives: {train_pos_cnt:,}, Negatives: {train_neg_cnt:,})", flush=True)
    print(f"Final Val Pair Count  : {len(val_pairs):,} (Positives: {val_pos_cnt:,}, Negatives: {val_neg_cnt:,})", flush=True)

    print("Extracting pairwise features for Training and Validation sets...", flush=True)
    start_feat_gen = time.time()
    X_train = extract_features_for_pairs(train_pairs, s1_dict, cand_dict, pair_reasons_map)
    y_train = np.array(train_labels, dtype=int)
    
    X_val = extract_features_for_pairs(val_pairs, s1_dict, cand_dict, pair_reasons_map)
    y_val = np.array(val_labels, dtype=int)
    feat_gen_runtime = time.time() - start_feat_gen
    print(f"Feature generation completed in {feat_gen_runtime:.2f}s (X_train shape = {X_train.shape}, X_val shape = {X_val.shape}).", flush=True)

    # 7. Model Selection & Comparison
    print("\n" + "="*80, flush=True)
    print("MODEL SELECTION & COMPARISON EVALUATION", flush=True)
    print("="*80, flush=True)

    models_config = {
        "LogisticRegression": {
            "matcher": EntityMatcher(model_type="logistic", C=1.0),
            "hyperparams": "C=1.0, max_iter=1000",
        },
        "HistGradientBoosting": {
            "matcher": EntityMatcher(model_type="hist_gb", n_estimators=200, learning_rate=0.05),
            "hyperparams": "n_estimators=200, learning_rate=0.05, max_depth=8",
        },
        "XGBoost": {
            "matcher": EntityMatcher(model_type="xgboost", n_estimators=300, learning_rate=0.05),
            "hyperparams": "n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8",
        },
        "RandomForest": {
            "matcher": EntityMatcher(model_type="random_forest", n_estimators=150),
            "hyperparams": "n_estimators=150, max_depth=15, n_jobs=-1",
        },
        "LightGBM": {
            "matcher": EntityMatcher(model_type="lightgbm", n_estimators=300, learning_rate=0.05, max_depth=8, num_leaves=31),
            "hyperparams": "n_estimators=300, learning_rate=0.05, max_depth=8, num_leaves=31",
        },
    }

    model_results = {}
    total_training_runtime = 0.0
    total_inference_runtime = 0.0
    
    for mname, cfg in models_config.items():
        matcher = cfg["matcher"]
        hparams = cfg["hyperparams"]
        
        print(f"\n--- Training {mname} ---", flush=True)
        t0 = time.time()
        matcher.fit(X_train, y_train)
        train_time = time.time() - t0
        total_training_runtime += train_time
        
        t1 = time.time()
        val_probs = matcher.predict_proba(X_val)
        infer_time = time.time() - t1
        total_inference_runtime += infer_time
        
        # Temp threshold 0.90 evaluation for model comparison
        preds = (val_probs >= 0.90).astype(int)
        tp = int(np.sum((preds == 1) & (y_val == 1)))
        fp = int(np.sum((preds == 1) & (y_val == 0)))
        fn = int(np.sum((preds == 0) & (y_val == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = calc_f05(prec, rec)

        print(f"{mname:<22} | Train Time: {train_time:5.2f}s | Infer Time: {infer_time:5.3f}s | Val F0.5 (@0.90): {f05*100:5.2f}% | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | TP={tp}, FP={fp}, FN={fn}", flush=True)

        model_results[mname] = {
            'matcher': matcher,
            'hyperparams': hparams,
            'val_probs': val_probs,
            'train_time': train_time,
            'infer_time': infer_time,
            'temp_f05': f05,
            'precision': prec,
            'recall': rec,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    # Best Model Selection
    best_mname = "LightGBM"
    val_probs = model_results[best_mname]['val_probs']

    # 8. Calibration Experiment
    print("\n" + "="*80, flush=True)
    print("PROBABILITY CALIBRATION EVALUATION", flush=True)
    print("="*80, flush=True)
    
    calib_matcher = EntityMatcher(model_type="lightgbm", calibrate=True, n_estimators=200, learning_rate=0.05)
    t0_cal = time.time()
    calib_matcher.fit(X_train, y_train)
    calib_train_time = time.time() - t0_cal
    calib_probs = calib_matcher.predict_proba(X_val)

    # Evaluate uncalibrated vs calibrated max F0.5
    raw_max_f05 = max([calc_f05(int(np.sum(((val_probs >= th).astype(int) == 1) & (y_val == 1))) / max(1, np.sum((val_probs >= th).astype(int) == 1)), int(np.sum(((val_probs >= th).astype(int) == 1) & (y_val == 1))) / max(1, np.sum(y_val == 1))) for th in np.linspace(0.1, 0.99, 20)])
    calib_max_f05 = max([calc_f05(int(np.sum(((calib_probs >= th).astype(int) == 1) & (y_val == 1))) / max(1, np.sum((calib_probs >= th).astype(int) == 1)), int(np.sum(((calib_probs >= th).astype(int) == 1) & (y_val == 1))) / max(1, np.sum(y_val == 1))) for th in np.linspace(0.1, 0.99, 20)])

    print(f"Raw Model Max Validation F0.5       : {raw_max_f05*100:6.2f}%", flush=True)
    print(f"Calibrated Model Max Validation F0.5 : {calib_max_f05*100:6.2f}%", flush=True)
    
    if calib_max_f05 > raw_max_f05:
        print("Adopting Calibrated Model for final predictions.", flush=True)
        use_matcher = calib_matcher
        val_probs = calib_probs
    else:
        print("Retaining Raw Model (calibration did not improve F0.5).", flush=True)
        use_matcher = model_results[best_mname]['matcher']

    # 9. Threshold Grid Search for Macro F0.5
    print("\n" + "="*80, flush=True)
    print("THRESHOLD GRID SEARCH FOR OPTIMAL MACRO F0.5", flush=True)
    print("="*80, flush=True)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    
    thresh_table = []
    best_f05 = -1.0
    best_thresh = 0.5
    best_metrics = {}

    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F0.5 Score':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Predicted Matches':<17}", flush=True)
    print("-" * 88, flush=True)

    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        tp = int(np.sum((preds == 1) & (y_val == 1)))
        fp = int(np.sum((preds == 1) & (y_val == 0)))
        fn = int(np.sum((preds == 0) & (y_val == 1)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = calc_f05(precision, recall)
        
        thresh_table.append({
            'threshold': th,
            'precision': precision,
            'recall': recall,
            'f05': f05,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'pred_cnt': tp + fp,
        })

        print(f"{th:<10.3f} | {precision*100:<9.2f}% | {recall*100:<9.2f}% | {f05*100:<9.2f}% | {tp:<6} | {fp:<6} | {fn:<6} | {tp+fp:<17,}", flush=True)

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th
            best_metrics = {
                'threshold': th,
                'precision': precision,
                'recall': recall,
                'f05': f05,
                'tp': tp,
                'fp': fp,
                'fn': fn,
            }

    print("-" * 88, flush=True)
    print(f"\nOPTIMAL THRESHOLD = {best_thresh:.3f} | Macro F0.5 = {best_f05*100:.2f}% | Precision = {best_metrics['precision']*100:.2f}% | Recall = {best_metrics['recall']*100:.2f}%", flush=True)

    # 10. Source Breakdown (S2 vs S3)
    val_s2_mask = np.array([cand_dict[c]['source_prefix'] == 's2' for _, c in val_pairs])
    val_s3_mask = np.array([cand_dict[c]['source_prefix'] == 's3' for _, c in val_pairs])

    def eval_mask(mask, name):
        p_m = val_probs[mask]
        y_m = y_val[mask]
        preds = (p_m >= best_thresh).astype(int)
        tp = int(np.sum((preds == 1) & (y_m == 1)))
        fp = int(np.sum((preds == 1) & (y_m == 0)))
        fn = int(np.sum((preds == 0) & (y_m == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = calc_f05(prec, rec)
        print(f"Source {name:<3} : Precision = {prec*100:6.2f}% | Recall = {rec*100:6.2f}% | Macro F0.5 = {f05*100:6.2f}% | TP={tp:,}, FP={fp:,}, FN={fn:,}", flush=True)
        return prec, rec, f05, tp, fp, fn

    print("\nSOURCE-SPECIFIC EVALUATION (At Optimal Threshold = {:.3f}):".format(best_thresh), flush=True)
    s2_prec, s2_rec, s2_f05, s2_tp, s2_fp, s2_fn = eval_mask(val_s2_mask, "S2")
    s3_prec, s3_rec, s3_f05, s3_tp, s3_fp, s3_fn = eval_mask(val_s3_mask, "S3")

    # 11. Singletons / No-Match Analysis
    print("\nSINGLETONS / NO-MATCH ENTITIES EVALUATION:", flush=True)
    val_s1_gt_no_match = [s for s in val_s1_set if len(s1_gt_map.get(s, set())) == 0]
    gt_no_match_cnt = len(val_s1_gt_no_match)
    
    pred_no_match_cnt = 0
    false_match_no_match_cnt = 0
    
    val_pair_probs = defaultdict(dict)
    for idx, (s1_id, cand_id) in enumerate(val_pairs):
        val_pair_probs[s1_id][cand_id] = val_probs[idx]

    for s1_id in val_s1_gt_no_match:
        c_probs = val_pair_probs[s1_id]
        matched = [c for c, p in c_probs.items() if p >= best_thresh]
        if len(matched) == 0:
            pred_no_match_cnt += 1
        else:
            false_match_no_match_cnt += 1

    false_match_rate = (false_match_no_match_cnt / gt_no_match_cnt * 100.0) if gt_no_match_cnt > 0 else 0.0
    print(f"  GT No-Match S1 Entities (Validation): {gt_no_match_cnt:,}", flush=True)
    print(f"  Correctly Predicted No-Match        : {pred_no_match_cnt:,}", flush=True)
    print(f"  False Match Rate on No-Match        : {false_match_rate:.2f}%", flush=True)

    # 12. Collect Error Samples (100+ FPs and 100+ FNs)
    print("\nCollecting Error Analysis Samples (100+ FPs, 100+ FNs)...", flush=True)
    val_preds = (val_probs >= best_thresh).astype(int)
    
    fp_samples = []
    fn_samples = []
    
    for idx, (s1_id, cand_id) in enumerate(val_pairs):
        pred_label = val_preds[idx]
        true_label = y_val[idx]
        prob = val_probs[idx]
        
        s1_rec = s1_dict[s1_id]
        cand_rec = cand_dict[cand_id]
        reasons = list(pair_reasons_map.get((s1_id, cand_id), set()))
        
        info = {
            's1_id': s1_id,
            's1_name': s1_rec['name_clean'],
            's1_address': s1_rec['address_clean'],
            's1_country': s1_rec['country'],
            'cand_id': cand_id,
            'cand_name': cand_rec['name_clean'],
            'cand_address': cand_rec['address_clean'],
            'cand_country': cand_rec['country'],
            'reasons': reasons,
            'prob': float(prob),
            'gt': int(true_label),
        }
        
        if pred_label == 1 and true_label == 0 and len(fp_samples) < 120:
            fp_samples.append(info)
        elif pred_label == 0 and true_label == 1 and len(fn_samples) < 120:
            fn_samples.append(info)

    # 13. Save Model and Evaluation Artifacts
    model_path = os.path.join(MODEL_DIR, "matcher_model.pkl")
    use_matcher.save(model_path)

    peak_mem_mb = get_peak_memory_mb()

    eval_artifacts = {
        'best_mname': best_mname,
        'best_thresh': best_thresh,
        'best_metrics': best_metrics,
        'thresh_table': thresh_table,
        'neg_results': neg_results,
        's1_count': len(s1_ids),
        's2_count': total_s2_records,
        's3_count': total_s3_records,
        'candidate_count': total_candidates_generated,
        'train_pos_cnt': train_pos_cnt,
        'train_neg_cnt': train_neg_cnt,
        'val_pos_cnt': val_pos_cnt,
        'val_neg_cnt': val_neg_cnt,
        'total_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'cand_gen_runtime': cand_gen_runtime,
        'feat_gen_runtime': feat_gen_runtime,
        'training_runtime': total_training_runtime,
        'inference_runtime': total_inference_runtime,
        'peak_memory_mb': peak_mem_mb,
        's2_prec': s2_prec,
        's2_rec': s2_rec,
        's2_f05': s2_f05,
        's2_tp': s2_tp,
        's2_fp': s2_fp,
        's2_fn': s2_fn,
        's3_prec': s3_prec,
        's3_rec': s3_rec,
        's3_f05': s3_f05,
        's3_tp': s3_tp,
        's3_fp': s3_fp,
        's3_fn': s3_fn,
        'gt_no_match_cnt': gt_no_match_cnt,
        'pred_no_match_cnt': pred_no_match_cnt,
        'false_match_rate': false_match_rate,
        'raw_max_f05': raw_max_f05,
        'calib_max_f05': calib_max_f05,
        'fp_samples': fp_samples,
        'fn_samples': fn_samples,
        'model_comparison': {
            m: {
                'hyperparams': model_results[m]['hyperparams'],
                'precision': model_results[m]['precision'],
                'recall': model_results[m]['recall'],
                'f05': model_results[m]['temp_f05'],
                'tp': model_results[m]['tp'],
                'fp': model_results[m]['fp'],
                'fn': model_results[m]['fn'],
                'train_time': model_results[m]['train_time'],
                'infer_time': model_results[m]['infer_time'],
            } for m in model_results
        }
    }
    
    with open(os.path.join(MODEL_DIR, "eval_artifacts.pkl"), "wb") as f:
        pickle.dump(eval_artifacts, f)

    print(f"\nPhase 3 Training Pipeline Completed Successfully in {time.time() - start_all:.2f}s!", flush=True)

if __name__ == "__main__":
    run_pipeline()

