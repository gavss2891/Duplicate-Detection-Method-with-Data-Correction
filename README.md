# CleanMSMP+ — Multi-Component Duplicate Detection with Data Correction

A scalable entity deduplication pipeline combining **Locality Sensitive Hashing (LSH)** with the **Multi-component Similarity Method (MSM)**. This repository introduces **CleanMSMP+**, an extended variant that applies structured data correction as a preprocessing step, and benchmarks it against the baseline **MSMP+**.

---

## The Problem: Duplicate Entities Across Heterogeneous Sources

Modern organizations ingest data from multiple sources — ERP systems, supplier portals, e-commerce feeds, procurement platforms, third-party data providers — and the same real-world entity often appears dozens of times under slightly different representations.

A TV listed as `"Samsung 55" QLED 4K"` on one retailer's website may appear as `"SAMSUNG QN55Q80C 55-Inch 4K Ultra HD"` on another. A vendor named `"Siemens AG"` in one procurement system may be recorded as `"SIEMENS Aktiengesellschaft"` or `"Siemens (Deutschland)"` in another. A spare part catalogued as `"Valve, Gate, DN50, PN16, CS"` in one system may be described as `"Gate Valve 2in 16bar Carbon Steel"` elsewhere.

These inconsistencies arise from free-text entry, varying standards, abbreviations, unit formatting differences, and the absence of a single master identifier. The result is **dirty master data**: inflated vendor lists, redundant material records, incorrect spend analytics, broken procurement workflows, and unreliable reporting.

Exact-match deduplication fails immediately in this setting. Purely string-based approaches such as edit distance do not generalize across the structured, multi-attribute nature of these records. A practical solution must reason about **multiple text components simultaneously** — weighing both feature-level key-value attributes and free-text descriptions — while remaining computationally tractable at scale.

---

## Where This Approach Applies

This pipeline is designed for any dataset where entities are described through **multiple text components** — a mix of structured attribute-value pairs and unstructured text fields. It has shown particular effectiveness in enterprise **Master Data Management (MDM)** scenarios:

### Procurement & Spend Analytics
Vendor and supplier master data in ERP systems (SAP, Oracle, Coupa) frequently contain hundreds or thousands of near-duplicate vendor records created by different business units, regions, or during system migrations. Deduplicating vendor masters directly improves spend consolidation, contract compliance, and supplier risk visibility.

### Materials & Parts Master
Manufacturing and MRO (Maintenance, Repair & Operations) catalogues accumulate duplicate material records with inconsistent descriptions, unit-of-measure variants, and partial specifications. Identifying that `"Bearing, Ball, 6205-2RS, 25x52x15mm"` and `"Ball Bearing 6205 RS 25mm bore"` refer to the same component is essential for accurate inventory management, purchasing efficiency, and obsolescence analysis.

### Customer Master Data
Customer records aggregated from CRM, billing, support, and marketing platforms routinely contain duplicate profiles for the same individual or organization — varying name formats, address abbreviations, and missing fields make exact matching unreliable.

### Product Information Management (PIM)
Retailers and distributors managing large product catalogs across multiple supplier feeds face the same challenge: the same SKU arrives with different titles, feature sets, and attribute formats. Deduplication is a prerequisite for catalog consolidation and accurate product matching.

### Healthcare Provider Directories
Provider records sourced from insurance networks, hospital systems, and licensing bodies frequently overlap, with name variations, credential abbreviations, and address inconsistencies preventing reliable matching.

### Financial Counterparty & Instrument Data
Reference data for legal entities (LEI, SWIFT BIC), financial instruments, and counterparties aggregated from multiple market data vendors contains substantial overlap under different naming conventions and identifier schemes.

---

## This Repository: A TV Product Dataset as a Worked Example

To demonstrate and benchmark the algorithm, we apply it to a publicly available dataset of **TV product listings** scraped from multiple e-commerce retailers. TVs are a natural fit: each listing carries a rich set of attribute-value pairs (screen size, resolution, refresh rate, HDR standard, connector types, etc.) alongside a free-text product title — precisely the multi-component structure found in enterprise master data.

The TV dataset is used here strictly as a benchmark vehicle. **The algorithm is not TV-specific.** Given any dataset of records with mixed structured attributes and text descriptions, the pipeline can be adapted by adjusting tokenization patterns for model words and feature key normalization to the domain at hand.

---

## How It Works

The pipeline eliminates the need for exhaustive O(n²) pairwise comparison through a two-stage design:

1. **Data Cleaning** — Normalize unit variants and remove noise. With CleanMSMP+, additionally apply structured data correction: round fractional values, fix malformed attribute entries, and strip retailer-specific filler tokens.
2. **LSH (Candidate Generation)** — Extract structured tokens (model words) from titles and feature maps. Compute MinHash signatures and apply LSH banding to hash records into buckets. Only records sharing a bucket become candidate pairs, reducing the comparison space from O(n²) to a tractable subset.
3. **MSM Scoring** — For each candidate pair, compute a composite similarity score combining: structured feature key-value similarity, model-word overlap from unmatched features, and title similarity via TMWMSim. Records from the same source or with conflicting brand tokens are immediately ruled out.
4. **Clustering** — Feed the resulting dissimilarity matrix into agglomerative clustering to group records into duplicate sets.
5. **Evaluation** — Measure performance across multiple bootstrap samples using Pair Completeness, Pair Quality, F1\*, Precision, Recall, and F1.

---

## Code Structure

### [lsh.py](lsh.py) — Candidate Pair Generation
Extracts structured tokens (model words) from product titles and feature maps, then uses MinHash signatures and LSH banding to efficiently find pairs of records likely to be duplicates. Products sharing a bucket become candidate pairs; all others are skipped, collapsing the comparison space dramatically.

### [MSM.py](MSM.py) — Similarity Scoring and Clustering
Computes a composite similarity score for each candidate pair across three signals: structured feature key-value similarity, model-word overlap from unmatched features, and title similarity from TMWMSim. Records from the same source or with conflicting brand signals are immediately excluded. The dissimilarity matrix is fed into agglomerative clustering.

### [TMWMSim.py](TMWMSim.py) — Title Similarity
Measures similarity between two record titles via their model words — the alphanumeric tokens that encode model numbers, dimensions, ratings, and identifiers. Handles conflict detection (same model word family but different values implies distinct entities) and combines cosine similarity with Levenshtein-based model word matching.

### [data_cleaning.py](data_cleaning.py) — Preprocessing and Data Correction
Normalizes raw records before similarity computation. Always normalizes unit variants (e.g. `inches`, `"`, `-inch` → `inch`). With data correction enabled (CleanMSMP+), additionally rounds fractional values, fixes malformed entries, and strips noise tokens such as retailer names and filler words. Also handles train/test splitting for bootstrap evaluation.

### [main.py](main.py) — Orchestration
Ties the full pipeline together: runs bootstrap evaluation, tunes parameters (γ, ε, μ, b) on the train split, evaluates on the test split, saves checkpoints so interrupted runs can be resumed, and produces comparison plots of CleanMSMP+ vs MSMP+.

### [analyze_data.py](analyze_data.py) — Dataset Inspection
Standalone utility for inspecting the raw dataset: lists sources, feature name frequencies, and records with duplicate identifiers within the same source. Not part of the detection pipeline.

---

## Installation

```bash
pip install numpy pandas scikit-learn sympy mmh3 ordered-set primePy python-Levenshtein matplotlib
```

The dataset file `TVs-all-merged.json` must be present in the working directory.

---

## Usage

### Basic Run

```bash
python main.py
```

Runs both CleanMSMP+ and MSMP+ across 8 bootstrap samples and saves comparison plots.

### Custom Parameters

```python
from main import main_func

main_func(
    path="TVs-all-merged.json",
    path_res=".",
    bootstraps=8,
    gammas=[0.5, 0.6, 0.7, 0.75],
    epsilons=[0.4, 0.5, 0.6],
    mus=[0.6, 0.65, 0.7],
    fraction=0.5,
    compare=True  # compare CleanMSMP+ vs MSMP+
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `gamma` (γ) | Similarity threshold for matching feature keys across records |
| `epsilon` (ε) | Clustering distance threshold |
| `mu` (μ) | Weight of title similarity in the composite MSM score |
| `fraction` | Controls the size of the MinHash signature matrix |
| `compare` | If `True`, runs both with and without data correction for benchmarking |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **F1** | Overall deduplication quality — harmonic mean of Precision and Recall |
| **F1\*** | Quality of LSH candidate generation — harmonic mean of PQ and PC |
| **PC** (Pair Completeness) | Share of true duplicate pairs captured as LSH candidates |
| **PQ** (Pair Quality) | Precision of LSH candidate pairs |
| **Fraction Comparisons** | Share of all possible pairs that LSH actually evaluated |

---

## Output

- **Plots** saved to `path_res/` comparing CleanMSMP+ vs MSMP+ across F1, F1\*, PC, PQ, Precision, and Recall as a function of Fraction of Comparisons.
- **Checkpoints** saved per bootstrap so interrupted runs resume automatically without recomputation.
- **Console output** with best and average metrics per bootstrap and across all runs.
