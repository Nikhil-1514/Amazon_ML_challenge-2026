# Dataset Profiling Report: Amazon ML Challenge 2026

## 1. Overview
The Amazon ML Challenge 2026 Business Entity Resolution dataset comprises **24,229,173 total entity records** split across training and test subsets:

| Subset | Source 1 (Reference) | Source 2 | Source 3 | Total Records |
|---|---|---|---|---|
| **Train** | 2,206,821 | 5,034,616 | 5,285,603 | **12,527,040** |
| **Test** | 1,732,544 | 4,887,273 | 5,082,316 | **11,702,133** |
| **Total** | **3,939,365** | **9,921,889** | **10,367,919** | **24,229,173** |

---

## 2. Key Attribute Profiling
- **Entity Names:** Highly heterogeneous text containing business legal suffixes (Inc, Corp, Pvt Ltd, SAS), multi-language variants, and special characters.
- **Addresses:** Street numbers, postal codes, locality tokens, and city-level names requiring standardized tokenization.
- **Country Distribution:** Primary coverage spans United States (`US`), India (`IN`), and France (`FR`), alongside an `OTHER` international subset.
- **Singletons & 1-to-Many Relationships:** In ground truth, ~14.3% of S1 entities are singletons (0 matches in S2/S3), while active business entities pair with 1 to 10+ branch records across S2 and S3.
