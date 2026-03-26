# TV Product Duplicate Detection

Duplicate detection system for TV products using Locality Sensitive Hashing (LSH) with MinHash signatures and the Multi-component Similarity Method (MSM). Includes an extended variant (**CleanMSMP+**) that applies data correction as a preprocessing step, compared against the baseline **MSMP+** approach.

## Overview

Given a dataset of TV product listings scraped from multiple e-commerce shops, the goal is to identify which listings refer to the same physical product (duplicates) without performing all O(n²) pairwise comparisons.

The pipeline:

1. **Data Cleaning** — Normalize units and remove noise; optionally apply extended data correction (CleanMSMP+)
2. **LSH** — Quickly narrow down the search space to a small set of candidate pairs likely to be duplicates
3. **MSM Scoring** — Compute a rich similarity score for each candidate pair using features and titles
4. **Clustering** — Group similar products into duplicate clusters
5. **Evaluation** — Measure performance across multiple bootstrap samples

## Code Structure

### `lsh.py` — Candidate Pair Generation
Extracts structured tokens (model words) from product titles and feature maps, then uses MinHash signatures and LSH banding to efficiently find pairs of products that are likely duplicates. Instead of comparing every product against every other, LSH hashes products into buckets — only products that share a bucket become candidate pairs. This reduces the number of comparisons from O(n²) to a manageable subset.

### `MSM.py` — Similarity Scoring and Clustering
For each candidate pair produced by LSH, computes a similarity score combining three signals: how similar the structured feature key-value pairs are, how much overlap there is in model words from unmatched features, and the title similarity from TMWMSim. Products from the same shop or with conflicting brands are immediately ruled out. The resulting dissimilarity matrix is fed into agglomerative clustering to group products into duplicate sets.

### `TMWMSim.py` — Title Similarity
Computes how similar two product titles are by looking at their model words — the alphanumeric tokens that typically encode screen size, resolution, model numbers, etc. It checks for conflicts (same model word family but different numbers implies different products) and combines cosine similarity with Levenshtein-based model word matching.

### `data_cleaning.py` — Preprocessing
Normalizes raw product data before any similarity computation. Always normalizes unit variants (e.g. `inches`, `"`, `-inch` → `inch`). With data correction enabled, additionally rounds fractional inch values, fixes malformed values, and strips noise tokens like retailer names and filler words. Also handles train/test splitting for bootstrap evaluation.

### `main.py` — Orchestration
Ties everything together: runs bootstrap evaluation, tunes parameters (γ, ε, μ, b) on a train split, evaluates on a test split, saves checkpoints so interrupted runs can be resumed, and plots the results comparing CleanMSMP+ vs MSMP+.

### `analyze_data.py` — Dataset Inspection
Standalone utility script (not part of the detection pipeline) to inspect the raw dataset — lists shops, feature name frequencies, and any model IDs with duplicate listings in the same store.

## Installation

```bash
pip install numpy pandas scikit-learn sympy mmh3 ordered-set primePy python-Levenshtein matplotlib
```

The dataset file `TVs-all-merged.json` must be present in the working directory.

## Usage

### Basic Run

```bash
python main.py
```

Runs both CleanMSMP+ and MSMP+ with 8 bootstrap samples and saves comparison plots.

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
| `gamma` (γ) | Threshold for matching feature keys by similarity |
| `epsilon` (ε) | Clustering distance threshold |
| `mu` (μ) | Weight of title similarity in the final MSM score |
| `fraction` | Controls the size of the MinHash signature |
| `compare` | If `True`, runs both with and without data correction for comparison |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **F1** | Overall duplicate detection quality (Precision/Recall harmonic mean) |
| **F1\*** | Quality of LSH candidate pairs (PQ/PC harmonic mean) |
| **PC** (Pair Completeness) | How many true duplicate pairs LSH captured as candidates |
| **PQ** (Pair Quality) | How precise the LSH candidate pairs are |
| **Fraction Comparisons** | What fraction of all possible pairs LSH actually compared |

## Output

- **Plots** saved to `path_res/` comparing CleanMSMP+ vs MSMP+ across F1, F1\*, PC, PQ, Precision, Recall vs. Fraction Comparisons.
- **Checkpoints** saved per bootstrap so interrupted runs can be resumed automatically.
- **Console output** with best and average metrics for each bootstrap and overall.
