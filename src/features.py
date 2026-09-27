import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

FEATURE_NAMES = [
    # Name Features (15)
    "name_exact",
    "core_name_exact",
    "name_jw",
    "name_lev_sim",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_token_jaccard",
    "name_token_overlap_coef",
    "name_token_containment",
    "name_common_token_cnt",
    "name_first_token_equal",
    "name_first_word_jw",
    "name_token_set_sim",
    "name_len_diff",
    "name_rel_len_diff",
    
    # Address Features (12)
    "addr_exact",
    "addr_token_jaccard",
    "addr_overlap_coef",
    "addr_char_3gram_jaccard",
    "addr_char_4gram_jaccard",
    "addr_jw",
    "addr_lev_sim",
    "addr_common_token_cnt",
    "addr_num_overlap_cnt",
    "addr_num_exact",
    "addr_locality_overlap_cnt",
    "rel_addr_len_diff",
    
    # Postal Features (4)
    "postal_exact",
    "postal_prefix_match",
    "postal_available_both",
    "postal_len_equal",
    
    # Country Features (1)
    "country_exact",
    
    # Cross-Field Interaction Features (6)
    "cross_name_and_addr_exact",
    "cross_name_high_and_addr_high",
    "cross_name_sim_x_addr_sim",
    "cross_name_sim_plus_addr_sim",
    "cross_postal_exact_and_name_sim",
    "cross_addr_num_and_name_sim",

    # Blocking Evidence Features (7)
    "block_exact_core_name",
    "block_postal",
    "block_name_token",
    "block_addr_token",
    "block_freq_token_composite",
    "block_first_name_word",
    "block_reason_cnt",
    
    # Source Features (2)
    "source_is_S2",
    "source_is_S3",
]

def get_ngram_jaccard(str1, str2, n=3):
    if not str1 or not str2:
        return 0.0
    if len(str1) < n or len(str2) < n:
        return 1.0 if str1 == str2 else 0.0
    
    ngrams1 = set(str1[i:i+n] for i in range(len(str1) - n + 1))
    ngrams2 = set(str2[i:i+n] for i in range(len(str2) - n + 1))
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return intersection / union if union > 0 else 0.0

def compute_pair_features(s1_rec, cand_rec, block_reasons=None):
    """
    Computes pairwise feature vector for one (S1, Candidate) pair.
    """
    n1 = s1_rec['name_clean']
    n2 = cand_rec['name_clean']
    
    c1 = s1_rec['core_name']
    c2 = cand_rec['core_name']
    
    a1 = s1_rec['address_clean']
    a2 = cand_rec['address_clean']
    
    p1 = s1_rec['postal_clean']
    p2 = cand_rec['postal_clean']
    
    ctry1 = s1_rec['country']
    ctry2 = cand_rec['country']
    
    ntoks1 = s1_rec['name_tokens_set']
    ntoks2 = cand_rec['name_tokens_set']
    
    atoks1 = s1_rec['addr_tokens_set']
    atoks2 = cand_rec['addr_tokens_set']
    
    anums1 = s1_rec['addr_nums_set']
    anums2 = cand_rec['addr_nums_set']
    
    # 1. Name Features
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    core_name_exact = 1.0 if (c1 and c1 == c2) else 0.0
    
    name_jw = JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0
    name_lev_dist = Levenshtein.distance(n1, n2) if (n1 and n2) else max(len(n1), len(n2))
    max_nlen = max(len(n1), len(n2), 1)
    name_lev_sim = 1.0 - (name_lev_dist / max_nlen)
    
    name_c3 = get_ngram_jaccard(n1, n2, n=3)
    name_c4 = get_ngram_jaccard(n1, n2, n=4)
    
    name_tok_inter = len(ntoks1.intersection(ntoks2))
    name_tok_union = len(ntoks1.union(ntoks2))
    name_tok_min = min(len(ntoks1), len(ntoks2))
    
    name_token_jaccard = name_tok_inter / name_tok_union if name_tok_union > 0 else 0.0
    name_token_overlap_coef = name_tok_inter / name_tok_min if name_tok_min > 0 else 0.0
    name_token_containment = name_tok_inter / len(ntoks1) if len(ntoks1) > 0 else 0.0
    name_common_token_cnt = float(name_tok_inter)
    
    w1 = s1_rec['first_word']
    w2 = cand_rec['first_word']
    name_first_token_equal = 1.0 if (w1 and w1 == w2) else 0.0
    name_first_word_jw = JaroWinkler.similarity(w1, w2) if (w1 and w2) else 0.0
    
    name_token_set_sim = float(len(ntoks1.intersection(ntoks2))) / max(len(ntoks1), len(ntoks2), 1)

    name_len_diff = float(abs(len(n1) - len(n2)))
    name_rel_len_diff = name_len_diff / max_nlen
    
    # 2. Address Features
    addr_exact = 1.0 if (a1 and a1 == a2) else 0.0
    
    addr_tok_inter = len(atoks1.intersection(atoks2))
    addr_tok_union = len(atoks1.union(atoks2))
    addr_tok_min = min(len(atoks1), len(atoks2))
    
    addr_token_jaccard = addr_tok_inter / addr_tok_union if addr_tok_union > 0 else 0.0
    addr_overlap_coef = addr_tok_inter / addr_tok_min if addr_tok_min > 0 else 0.0
    addr_char_3gram_jaccard = get_ngram_jaccard(a1, a2, n=3)
    addr_char_4gram_jaccard = get_ngram_jaccard(a1, a2, n=4)

    addr_jw = JaroWinkler.similarity(a1, a2) if (a1 and a2) else 0.0
    addr_lev_dist = Levenshtein.distance(a1, a2) if (a1 and a2) else max(len(a1), len(a2))
    max_alen = max(len(a1), len(a2), 1)
    addr_lev_sim = 1.0 - (addr_lev_dist / max_alen)
    
    addr_common_token_cnt = float(addr_tok_inter)
    
    num_inter = len(anums1.intersection(anums2))
    addr_num_overlap_cnt = float(num_inter)
    addr_num_exact = 1.0 if (len(anums1) > 0 and anums1 == anums2) else 0.0

    loc1 = s1_rec.get('location_tokens_set', set())
    loc2 = cand_rec.get('location_tokens_set', set())
    addr_locality_overlap_cnt = float(len(loc1.intersection(loc2)))
    
    rel_addr_len_diff = float(abs(len(a1) - len(a2))) / max_alen
    
    # 3. Postal Features
    postal_exact = 1.0 if (p1 and p1 == p2) else 0.0
    postal_prefix_match = 1.0 if (p1 and p2 and p1[:3] == p2[:3]) else 0.0
    postal_available_both = 1.0 if (p1 and p2) else 0.0
    postal_len_equal = 1.0 if (p1 and p2 and len(p1) == len(p2)) else 0.0
    
    # 4. Country Feature
    country_exact = 1.0 if (ctry1 and ctry1 == ctry2) else 0.0
    
    # 5. Cross-Field Interaction Features
    cross_name_and_addr_exact = name_exact * addr_exact
    cross_name_high_and_addr_high = 1.0 if (name_jw >= 0.85 and addr_jw >= 0.85) else 0.0
    cross_name_sim_x_addr_sim = name_jw * addr_jw
    cross_name_sim_plus_addr_sim = name_jw + addr_jw
    cross_postal_exact_and_name_sim = postal_exact * name_jw
    cross_addr_num_and_name_sim = (1.0 if num_inter > 0 else 0.0) * name_jw
    
    # 6. Blocking Evidence Features
    reasons = block_reasons or set()
    block_exact_core_name = 1.0 if "EXACT_CORE_NAME" in reasons else 0.0
    block_postal = 1.0 if "POSTAL_CODE" in reasons else 0.0
    block_name_token = 1.0 if "NAME_TOKEN" in reasons else 0.0
    block_addr_token = 1.0 if "ADDRESS_TOKEN" in reasons else 0.0
    block_freq_token_composite = 1.0 if "FREQ_TOKEN_NUMERIC_COMPOSITE" in reasons else 0.0
    block_first_name_word = 1.0 if "FIRST_NAME_WORD" in reasons else 0.0
    block_reason_cnt = float(len(reasons))

    # 7. Source Features
    source_prefix = cand_rec.get('source_prefix', '')
    source_is_S2 = 1.0 if source_prefix.lower() == 's2' else 0.0
    source_is_S3 = 1.0 if source_prefix.lower() == 's3' else 0.0
    
    return [
        name_exact,
        core_name_exact,
        name_jw,
        name_lev_sim,
        name_c3,
        name_c4,
        name_token_jaccard,
        name_token_overlap_coef,
        name_token_containment,
        name_common_token_cnt,
        name_first_token_equal,
        name_first_word_jw,
        name_token_set_sim,
        name_len_diff,
        name_rel_len_diff,
        
        addr_exact,
        addr_token_jaccard,
        addr_overlap_coef,
        addr_char_3gram_jaccard,
        addr_char_4gram_jaccard,
        addr_jw,
        addr_lev_sim,
        addr_common_token_cnt,
        addr_num_overlap_cnt,
        addr_num_exact,
        addr_locality_overlap_cnt,
        rel_addr_len_diff,
        
        postal_exact,
        postal_prefix_match,
        postal_available_both,
        postal_len_equal,
        
        country_exact,
        
        cross_name_and_addr_exact,
        cross_name_high_and_addr_high,
        cross_name_sim_x_addr_sim,
        cross_name_sim_plus_addr_sim,
        cross_postal_exact_and_name_sim,
        cross_addr_num_and_name_sim,

        block_exact_core_name,
        block_postal,
        block_name_token,
        block_addr_token,
        block_freq_token_composite,
        block_first_name_word,
        block_reason_cnt,
        
        source_is_S2,
        source_is_S3,
    ]

def extract_features_for_pairs(pairs, s1_dict, cand_dict, pair_reasons_map=None):
    """
    Extracts numpy feature array (N_pairs, N_features) given:
    pairs: list of tuples (s1_id, cand_id)
    s1_dict: map of s1_id -> precomputed record dict
    cand_dict: map of cand_id -> precomputed record dict
    pair_reasons_map: optional dict mapping (s1_id, cand_id) -> set of blocking reasons
    """
    X = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(pairs):
        s1_rec = s1_dict[s1_id]
        cand_rec = cand_dict[cand_id]
        reasons = pair_reasons_map.get((s1_id, cand_id), set()) if pair_reasons_map else set()
        X[i] = compute_pair_features(s1_rec, cand_rec, block_reasons=reasons)
    return X
