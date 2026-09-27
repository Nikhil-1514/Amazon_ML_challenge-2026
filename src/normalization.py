import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES_PAT = re.compile(
    r'\b(corp|corporation|ltd|limited|pvt|private|llc|inc|incorporated|co|company|gmbh|sa|sarl|plc|llp)\b',
    re.IGNORECASE
)

def clean_str(val):
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()

def normalize_text(text):
    text = clean_str(text)
    if not text:
        return ""
    # Unicode normalization (NFKD)
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    # Normalize & to and
    text = re.sub(r'&', ' and ', text)
    # Remove punctuation / non-alphanumeric (keep whitespace)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pin(address):
    address = clean_str(address)
    if not address:
        return None
    # Indian 6 digit pin
    match_in = re.search(r'\b[1-9][0-9]{5}\b', address)
    if match_in:
        return match_in.group(0)
    # US 5 digit zip code
    match_us = re.search(r'\b[0-9]{5}(?:-[0-9]{4})?\b', address)
    if match_us:
        return match_us.group(0)[:5]
    return None

def extract_legal_suffixes(name):
    norm_name = normalize_text(name)
    if not norm_name:
        return "", "", []
    
    suffixes_found = LEGAL_SUFFIXES_PAT.findall(norm_name)
    # Strip trailing legal suffixes to form normalized_core_name
    core_name = LEGAL_SUFFIXES_PAT.sub(' ', norm_name)
    core_name = re.sub(r'\s+', ' ', core_name).strip()
    
    if not core_name:
        core_name = norm_name
        
    return norm_name, core_name, [s.lower() for s in suffixes_found]

def tokenize_name(name, min_len=4):
    norm_name = normalize_text(name)
    if not norm_name:
        return []
    tokens = norm_name.split()
    # Filter tokens by length
    return [t for t in tokens if len(t) >= min_len]

def tokenize_address(address, min_len=4):
    norm_addr = normalize_text(address)
    if not norm_addr:
        return []
    tokens = norm_addr.split()
    addr_tokens = []
    for t in tokens:
        # Include words >= min_len or numeric tokens (e.g. street numbers, building numbers)
        if len(t) >= min_len or t.isdigit():
            addr_tokens.append(t)
    return addr_tokens

def extract_location_tokens(address, country=""):
    norm_addr = normalize_text(address)
    if not norm_addr:
        return set()
    tokens = set(norm_addr.split())
    # Return tokens that look like locations (alphanumeric, length >= 3)
    return {t for t in tokens if len(t) >= 3 and not t.isdigit()}

def extract_addr_numerics(address):
    address = clean_str(address)
    if not address:
        return []
    nums = re.findall(r'\b\d{1,6}\b', address)
    return list(set(nums))

