# Fashion Search Ranking (Code‑only)

This repository shares code only. No datasets are included due to data privacy. The pipeline processes Parquet logs with Polars, engineers features, trains LightGBM rankers, and produces a submission.

## What this contains
- Main script: `Proje.py`
- No data files under `data/` are provided.

## Requirements
- Python 3.10+
- Packages: `polars`, `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `scipy`

Install:
- `pip install polars numpy pandas scikit-learn lightgbm scipy`

## Data privacy
- Datasets are not distributed in this repo.
- To run locally, place your own Parquet files under `data/` or update `DATA_PATH` inside `Proje.py`.

## Expected layout (user‑provided)
- Root files:
  - `data/train_sessions.parquet`
  - `data/test_sessions.parquet`
- Content:
  - `data/content/metadata.parquet`
  - `data/content/price_rate_review_data.parquet`
  - `data/content/search_log.parquet`
  - `data/content/sitewide_log.parquet`
  - `data/content/top_terms_log.parquet`
- User:
  - `data/user/metadata.parquet`
  - `data/user/sitewide_log.parquet`
  - `data/user/search_log.parquet`
  - `data/user/top_terms_log.parquet`
  - `data/user/fashion_search_log.parquet`
  - `data/user/fashion_sitewide_log.parquet`
- Term:
  - `data/term/search_log.parquet`

## Key columns expected
- Sessions: `session_id`, `user_id_hashed`, `content_id_hashed`, `ts_hour`, `search_term_normalized`, `clicked`, `ordered`, `added_to_cart`, `added_to_fav`
- Aggregates: `total_search_impression`, `total_search_click`, `total_click`, `total_order`, `total_cart`, `total_fav`
- Content meta: `level1_category_name`, `level2_category_name`, `leaf_category_name`, `attribute_type_count`, `total_attribute_option_count`, `merchant_count`, `filterable_label_count`, `content_creation_date`, `cv_tags`
- Price/rating: `update_date`, `original_price`, `selling_price`, `discounted_price`, `content_review_count`, `content_review_wth_media_count`, `content_rate_count`, `content_rate_avg`

## Configuration
- Update base path in `Proje.py`: `DATA_PATH = '.../data'`
- Random seed: `SEED = 42`
- Time‑based validation is used. If timestamps are missing, the script falls back to a small random validation set.

## Run
- `python Proje.py`

## Output
- Submission file: `submission14.csv` with `session_id` and space‑separated `content_id_hashed` predictions.

## Notes
- Works on Windows paths and Unicode. Uses Polars for streaming/grouped aggregations and LightGBM `LGBMRanker` for ranking.
- Feature lists and joins are guarded; missing optional sources are skipped.
