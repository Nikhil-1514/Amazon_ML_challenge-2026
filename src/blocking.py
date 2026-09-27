import os
import time
from collections import defaultdict, Counter
import pandas as pd
from src.normalization import (
    clean_str,
    normalize_text,
    extract_pin,
    extract_legal_suffixes,
    tokenize_name,
    tokenize_address,
)

DEFAULT_CONFIG = {
    "min_name_token_len": 4,
    "min_addr_token_len": 4,
    "max_df_name_token": 10000,
    "max_df_addr_token": 10000,
    "max_candidates_per_s1": None,  # None = Uncapped by default; can be set for candidate capping
}

class BlockingIndexer:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Inverted indexes: (country, token) -> list of entity IDs
        self.name_token_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.pin_index = defaultdict(list)
        self.core_name_index = defaultdict(list)
        self.prefix_index = defaultdict(list)
        self.first_word_index = defaultdict(list)
        
        self.addr_num_index = defaultdict(list)
        
        # S2 and S3 ID sets for validation
        self.indexed_s2_ids = set()
        self.indexed_s3_ids = set()
        
    def build_index(self, s2_file_path, s3_file_path, chunk_size=100000, max_rows=None):
        print("Building inverted blocking indexes from Source 2 and Source 3...")
        start_time = time.time()
        
        self.name_token_index.clear()
        self.addr_token_index.clear()
        self.pin_index.clear()
        self.core_name_index.clear()
        self.prefix_index.clear()
        self.first_word_index.clear()
        self.addr_num_index.clear()
        self.indexed_s2_ids.clear()
        self.indexed_s3_ids.clear()
        
        min_n_len = self.config["min_name_token_len"]
        min_a_len = self.config["min_addr_token_len"]
        
        total_indexed_records = 0
        
        for file_path, id_set in [(s2_file_path, self.indexed_s2_ids), (s3_file_path, self.indexed_s3_ids)]:
            print(f"Indexing {os.path.basename(file_path)}...", flush=True)
            rows_done = 0
            for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size, dtype=str, keep_default_na=False):
                if max_rows and rows_done >= max_rows:
                    break
                if max_rows and (rows_done + len(chunk) > max_rows):
                    chunk = chunk.iloc[:max_rows - rows_done]
                    
                rows_done += len(chunk)
                total_indexed_records += len(chunk)
                
                for row in chunk.itertuples(index=False):
                    eid = row.entity_id
                    bname = row.business_name
                    baddr = row.business_address
                    country = clean_str(row.country).upper()
                    
                    if not country:
                        continue
                        
                    id_set.add(eid)
                    
                    # 1. Name tokens & Core Name & Prefix
                    norm_name, core_name, _ = extract_legal_suffixes(bname)
                    if norm_name:
                        if core_name:
                            self.core_name_index[(country, core_name)].append(eid)
                            
                        words = norm_name.split()
                        if words:
                            self.first_word_index[(country, words[0])].append(eid)
                            
                        pref3 = norm_name[:3] if len(norm_name) >= 3 else norm_name
                        if pref3:
                            self.prefix_index[(country, pref3)].append(eid)
                            
                        n_tokens = tokenize_name(bname, min_len=min_n_len)
                        for tok in set(n_tokens):
                            self.name_token_index[(country, tok)].append(eid)
                            
                    # 2. Address tokens & PIN & Numerics
                    if baddr:
                        pin = extract_pin(baddr)
                        if pin:
                            self.pin_index[(country, pin)].append(eid)
                            
                        a_tokens = tokenize_address(baddr, min_len=min_a_len)
                        for tok in set(a_tokens):
                            self.addr_token_index[(country, tok)].append(eid)

                        from src.normalization import extract_addr_numerics
                        nums = extract_addr_numerics(baddr)
                        for num in nums:
                            self.addr_num_index[(country, num)].append(eid)
                            
                print(f"  [{os.path.basename(file_path)}] Processed {rows_done:,} rows...", flush=True)
                             
        # Convert index list values to sets for 100x faster C-level set operations
        print("Converting list indices to sets...", flush=True)
        for idx_map in [
            self.name_token_index,
            self.addr_token_index,
            self.pin_index,
            self.core_name_index,
            self.prefix_index,
            self.first_word_index,
            self.addr_num_index,
        ]:
            for k in list(idx_map.keys()):
                idx_map[k] = set(idx_map[k])

        elapsed = time.time() - start_time
        print(f"Index built successfully in {elapsed:.2f}s ({total_indexed_records:,} total records indexed).", flush=True)

    def get_candidate_matches_for_s1(self, bname, baddr, country, strategies=None):
        country = clean_str(country).upper()
        if not country:
            return {}
            
        if strategies is None:
            strategies = ["EXACT_CORE_NAME", "NAME_TOKEN", "ADDRESS_TOKEN", "POSTAL_CODE"]
            
        candidates = defaultdict(set)
        
        min_n_len = self.config["min_name_token_len"]
        min_a_len = self.config["min_addr_token_len"]
        max_df_name = self.config.get("max_df_name_token", 10000)
        max_df_addr = self.config.get("max_df_addr_token", 10000)
        
        norm_name, core_name, _ = extract_legal_suffixes(bname)
        
        # Strategy A: EXACT_CORE_NAME
        if "EXACT_CORE_NAME" in strategies and core_name:
            eids = self.core_name_index.get((country, core_name), [])
            for eid in eids:
                candidates[eid].add("EXACT_CORE_NAME")
                
        # Strategy B: FIRST_NAME_WORD
        if "FIRST_NAME_WORD" in strategies and norm_name:
            words = norm_name.split()
            if words:
                eids = self.first_word_index.get((country, words[0]), [])
                for eid in eids:
                    candidates[eid].add("FIRST_NAME_WORD")
                    
        # Strategy C: NAME_PREFIX_3
        if "NAME_PREFIX_3" in strategies and norm_name:
            pref3 = norm_name[:3] if len(norm_name) >= 3 else norm_name
            if pref3:
                eids = self.prefix_index.get((country, pref3), [])
                for eid in eids:
                    candidates[eid].add("NAME_PREFIX_3")
                    
        # Strategy D: NAME_TOKEN + VAR5 COMPOSITE
        if "NAME_TOKEN" in strategies:
            n_tokens = tokenize_name(bname, min_len=min_n_len)
            from src.normalization import extract_addr_numerics
            addr_nums = extract_addr_numerics(baddr) if baddr else []

            for tok in set(n_tokens):
                eids = self.name_token_index.get((country, tok), [])
                df = len(eids)
                if df == 0:
                    continue
                if max_df_name and df <= max_df_name:
                    for eid in eids:
                        candidates[eid].add("NAME_TOKEN")
                elif df <= 50000 and addr_nums:
                    for num in addr_nums:
                        num_eids = self.addr_num_index.get((country, num), set())
                        if num_eids:
                            intersect = eids.intersection(num_eids)
                            for eid in intersect:
                                candidates[eid].add("FREQ_TOKEN_NUMERIC_COMPOSITE")

        # Strategy E: ADDRESS_TOKEN
        if "ADDRESS_TOKEN" in strategies and baddr:
            a_tokens = tokenize_address(baddr, min_len=min_a_len)
            for tok in set(a_tokens):
                eids = self.addr_token_index.get((country, tok), [])
                if max_df_addr and len(eids) > max_df_addr:
                    continue
                for eid in eids:
                    candidates[eid].add("ADDRESS_TOKEN")
                    
        # Strategy F: POSTAL_CODE
        if "POSTAL_CODE" in strategies and baddr:
            pin = extract_pin(baddr)
            if pin:
                eids = self.pin_index.get((country, pin), [])
                for eid in eids:
                    candidates[eid].add("POSTAL_CODE")
                    
        max_cands = self.config.get("max_candidates_per_s1")
        if max_cands and len(candidates) > max_cands:
            sorted_cands = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)
            candidates = dict(sorted_cands[:max_cands])
            
        return candidates

