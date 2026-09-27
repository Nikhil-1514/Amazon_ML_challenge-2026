import os
import sys

DELIM = "\t"

def validate_candidate_pairs(candidate_path, test_dir):
    print("Validating candidate_pairs.tsv structure and format...")
    errors = []
    warnings = []
    
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    if not os.path.exists(test_s1_path):
        errors.append(f"test_source1.tsv not found at {test_s1_path}")
        return errors, warnings
        
    # Read required S1 IDs
    required_s1 = set()
    with open(test_s1_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if line.strip():
                required_s1.add(line.split(DELIM, 1)[0].strip())
                
    if not os.path.exists(candidate_path):
        errors.append(f"Candidate file not found: {candidate_path}")
        return errors, warnings
        
    seen_s1 = set()
    dup_s1 = set()
    self_matches = set()
    malformed_prefixes = set()
    intra_dupes = set()
    total_rows = 0
    empty_rows = 0
    total_cands = 0
    
    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append("candidate_pairs.tsv is empty.")
            return errors, warnings
            
        cols = [c.strip() for c in header.rstrip("\n").split(DELIM)]
        if cols != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"Invalid header: {cols}. Expected ['source1_entity_id', 'candidate_entity_ids']")
            return errors, warnings
            
        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) < 1:
                continue
            s1_id = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            
            if s1_id in seen_s1:
                dup_s1.add(s1_id)
            seen_s1.add(s1_id)
            total_rows += 1
            
            if not rest:
                empty_rows += 1
            else:
                c_ids = [c.strip() for c in rest.split(",") if c.strip()]
                if len(c_ids) != len(set(c_ids)):
                    intra_dupes.add(s1_id)
                total_cands += len(c_ids)
                
                for cid in c_ids:
                    if cid.startswith("S1-"):
                        self_matches.add(cid)
                    elif not cid.startswith(("S2-", "S3-")):
                        malformed_prefixes.add(cid)

    missing_s1 = required_s1 - seen_s1
    extra_s1 = seen_s1 - required_s1
    
    if dup_s1:
        errors.append(f"Duplicate S1 rows found: {len(dup_s1)} duplicate IDs")
    if intra_dupes:
        errors.append(f"Duplicate candidate IDs within a row: {len(intra_dupes)} rows")
    if self_matches:
        errors.append(f"Self-matches (S1- prefix in candidate IDs): {len(self_matches)} IDs")
    if malformed_prefixes:
        errors.append(f"Malformed candidate ID prefixes: {len(malformed_prefixes)} IDs")
    if missing_s1:
        errors.append(f"Missing required S1 entities: {len(missing_s1)} entities")
    if extra_s1:
        errors.append(f"Unknown S1 entity IDs not in test_source1: {len(extra_s1)} entities")
        
    print(f"Validation summary for {os.path.basename(candidate_path)}:")
    print(f"  Total S1 rows: {total_rows:,}")
    print(f"  Empty candidate rows: {empty_rows:,} ({empty_rows/total_rows*100:.2f}%)")
    print(f"  Total candidate pairs: {total_cands:,}")
    print(f"  Average candidates per S1: {total_cands/total_rows:.2f}")
    
    if errors:
        print("FAIL — Validation issues found:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("PASS — candidate_pairs.tsv passed 100% of formatting and structural rules!")
        
    return errors, warnings

if __name__ == "__main__":
    cand_p = sys.argv[1] if len(sys.argv) > 1 else "output/candidate_pairs.tsv"
    test_d = sys.argv[2] if len(sys.argv) > 2 else "dataset/test"
    validate_candidate_pairs(cand_p, test_d)
