# LightGCN+ Book Recommendation System

## Overview

Graph-based book recommendation system built on the [Book-Crossing dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). Extends LightGCN (He et al., SIGIR 2020) with cold start handling, rating-weighted graph edges, implicit feedback integration, and residual skip connections.

**Key contribution:** Unlike standard approaches that filter out users with few ratings, this model explicitly handles cold users (as few as 3 ratings) through a two-regime architecture — making recommendations for 4,576 cold users that baseline methods ignore entirely.

## Results

Evaluated using 99-sampled negatives (same protocol as NeuMF baseline):

| Metric | NeuMF (baseline) | LightGCN+ v4 (ours) |
|--------|-----------------|---------------------|
| HR@5   | 0.268 | **0.352** |
| HR@10  | 0.440 | **0.463** |
| HR@20  | 0.590 | **0.596** |
| NDCG@5 | 0.157 | **0.263** |
| NDCG@10| 0.213 | **0.299** |
| NDCG@20| 0.251 | **0.333** |

### Cold Start Results

| Segment | n users | HR@10 | NDCG@10 |
|---------|---------|-------|---------|
| Cold (≤10 train ratings) | 4,576 | 0.403 | 0.265 |
| Warm (>10 train ratings) | 7,414 | 0.500 | 0.320 |

## Dataset

| | LightGCN+ v4 | NeuMF |
|--|--|--|
| Users | 7,025 | ~3,300 |
| Books | 9,432 | ~3,300 |
| Explicit ratings | 118,668 | ~33,000 |
| Implicit interactions | 178,652 | 0 |
| Cold users included | 4,576 | 0 |
| Min user ratings | 5 | 20 |
| Filtering | Converged (19 passes) | 3-pass only |

## Model Architecture

- **Base:** LightGCN with 4-layer graph propagation
- **Graph edges:** Explicit (weight = rating/10) + Implicit (weight = 0.1)
- **Residual connections:** E(k) = A_hat @ E(k-1) + 0.2 * E(0) — prevents over-smoothing
- **Cold start:** Feature-only regime for users with ≤10 training ratings
- **Features:** Author, publisher, decade (books) | Age, country (users)
- **Scoring:** Cosine similarity (L2 normalized embeddings)
- **Loss:** Rating-weighted BPR with popularity-weighted → semi-hard negative mining
- **Hardware:** Optimized for Apple M4 Pro (MPS + CPU sparse ops)

## Repo Structure

```
LightGCN-BookCrossing/
├── notebooks/
│   └── LightGCN_v4.ipynb       # Full training + evaluation notebook
├── results/
│   ├── results.json            # All metrics (val + test + cold/warm breakdown)
│   ├── lightgcn_v4_best.pt     # Best model weights
│   └── error_analysis/
│       ├── report.json         # Summary report
│       ├── diagnostics.csv     # Per-user diagnostic data
│       ├── 01_rank_cdf.png
│       ├── 02_cold_vs_warm.png
│       ├── 03_activity_segments.png
│       ├── 04_popularity_bias.png
│       ├── 05_score_calibration.png
│       ├── 06_heatmap.png
│       ├── 07_training_curve.png
│       ├── 08_version_comparison.png
│       └── 09_sampled_vs_full.png
└── reference/
    └── LightGCN_Reference.pdf  # One-page group reference sheet
```

## How to Run

1. Download the Book-Crossing dataset (3 CSV files):
   - `BX-Book-Ratings.csv`
   - `BX-Users.csv`
   - `BX-Books.csv`

2. Place all 3 CSVs in the same folder as the notebook

3. Update `DATA_DIR` in Cell 3 to point to that folder

4. Run all cells top to bottom

**Requirements:** Python 3.10+, PyTorch, pandas, scipy, scikit-learn, matplotlib, seaborn

## References

- He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020
- He et al., "Neural Collaborative Filtering", WWW 2017
- Ziegler et al., "Improving Recommendation Lists Through Topic Diversification", WWW 2005 (Book-Crossing dataset)
