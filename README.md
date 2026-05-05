# CleanMSMP+ — Duplicate Detection with Data Correction

This project builds on **MSMP+**, a duplicate detection pipeline that uses **LSH** for candidate generation and **MSM** for similarity scoring. The extension, **CleanMSMP+**, adds a structured data correction step before everything else runs. Evaluation against benchmarks show it improves overall model performance.  

> **This algorithm has been proven effective in production — running on 200,000+ material records in a real procurement environment.**
> The TV dataset here is just a toy demo to benchmark the approach.

---

## What is MSM?

**MSM (Multi-component Similarity Method)** is the core of the pipeline. It's designed for records that carry both structured attributes (key-value pairs) and free-text descriptions, which is exactly how real-world master data looks.

Instead of treating a record as a single string, MSM breaks it into components and scores similarity across three signals simultaneously:
- **Structured feature similarity** — how well the key-value attributes align across records
- **Model-word overlap** — alphanumeric tokens (model numbers, dimensions, specs) extracted from unmatched features
- **Title similarity (TMWMSim)** — model-word-aware title comparison that handles conflicts (same family, different values = different entity)

This makes it robust to the kinds of variation that kill simpler approaches: inconsistent attribute naming, missing fields, partial descriptions, and mixed structured/unstructured content.

### Where it applies

MSM works on any dataset where entities are described through multiple text components. In practice:

- **Materials & MRO catalogues** — spot that `"Bearing, Ball, 6205-2RS, 25x52x15mm"` and `"Ball Bearing 6205 RS 25mm bore"` are the same part
- **Vendor/supplier master data** — consolidate `"Siemens AG"` and `"SIEMENS Aktiengesellschaft"` across ERP systems
- **Product catalogs (PIM)** — match the same SKU arriving from different suppliers with different titles and attribute formats
- **Customer master data** — deduplicate CRM, billing, and support profiles for the same person or org
- **Healthcare provider directories** — reconcile provider records across insurance networks and hospital systems

---

## Scalability & Customization

MSM is highly customizable depending on the deduplication task. The comparison space can be reduced in two ways:

**LSH** generates candidate pairs via MinHash banding so you never do O(n²) comparisons — only records that hash into the same bucket get compared.

**Strict grouping criteria** can reduce running time even further. If your data has a natural partition where cross-group duplicates are impossible, you only need to run comparisons within each group. For example, in material master deduplication, two materials with different commodity group labels are guaranteed not to be duplicates — so similarity can be computed within each commodity group only, cutting runtime substantially on large datasets.

---

## How It Works

1. **Data Cleaning** — Normalize unit variants. CleanMSMP+ also rounds fractional values, fixes malformed entries, and strips retailer-specific noise.
2. **LSH** — Extract model words from titles and feature maps, compute MinHash signatures, and band them into buckets. Only records sharing a bucket become candidate pairs.
3. **MSM Scoring** — Compute a composite similarity score for each candidate pair across structured features, model-word overlap, and title similarity. Pairs from the same source or with conflicting brand tokens are dropped immediately.
4. **Clustering** — Feed the dissimilarity matrix into agglomerative clustering to group duplicates.
5. **Evaluation** — Measure performance across bootstrap samples using F1, F1\*, PC, PQ, Precision, and Recall.

---

## Code Structure

| File | What it does |
|------|-------------|
| [lsh.py](lsh.py) | MinHash + LSH banding to generate candidate pairs |
| [MSM.py](MSM.py) | Similarity scoring and agglomerative clustering |
| [TMWMSim.py](TMWMSim.py) | Title similarity via model word matching |
| [data_cleaning.py](data_cleaning.py) | Normalization and (optionally) structured data correction |
| [main.py](main.py) | Orchestrates everything: bootstrap runs, parameter tuning, checkpointing, plots |
| [analyze_data.py](analyze_data.py) | Standalone utility for inspecting the raw dataset |

---

## Key Parameters

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
