import polars as pl
import numpy as np
import lightgbm as lgb
from datetime import datetime
import gc
import os
import multiprocessing

os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
from sklearn.metrics import ndcg_score, roc_auc_score
from itertools import product

DATA_PATH = "C:\\Users\\kağan burhan\\PycharmProjects\\Proje\\data"
SEED = 42


def clean_memory():
    gc.collect()


def load_and_aggregate_streaming(file_path: str, group_cols: list, agg_exprs: list, chunk_size=500000):
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return pl.DataFrame()

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"🔄 Processing {os.path.basename(file_path)} ({file_size_mb:.1f} MB)")

        if file_size_mb < 100:
            result = pl.scan_parquet(file_path).group_by(group_cols).agg(agg_exprs).collect()
            print(f"✅ {os.path.basename(file_path)} completed (small file)")
            return result

        scan = pl.scan_parquet(file_path)
        total_rows = scan.select(pl.len()).collect().item()
        print(f"   Total rows: {total_rows:,}")

        agg_dict = {}
        for start in range(0, total_rows, chunk_size):
            chunk = scan.slice(start, chunk_size).collect()
            if chunk.is_empty(): break

            chunk_agg = chunk.group_by(group_cols).agg(agg_exprs)
            for row in chunk_agg.iter_rows(named=True):
                key = tuple(row[col] for col in group_cols)
                if key not in agg_dict:
                    agg_dict[key] = {col: 0 for col in chunk_agg.columns if col not in group_cols}
                for col in chunk_agg.columns:
                    if col not in group_cols:
                        agg_dict[key][col] += row[col] or 0

            del chunk, chunk_agg
            if start % (chunk_size * 5) == 0: clean_memory()

        if not agg_dict: return pl.DataFrame()

        result_data = []
        for key, values in agg_dict.items():
            row_dict = dict(zip(group_cols, key))
            row_dict.update(values)
            result_data.append(row_dict)

        result = pl.DataFrame(result_data)
        print(f"✅ {os.path.basename(file_path)} completed - {len(result):,} unique groups")
        clean_memory()
        return result
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
        return pl.DataFrame()


def smart_process_fashion_search(file_path: str, sessions_df, agg_exprs: list, sample_rate, chunk_size=250000):
    try:
        if not os.path.exists(file_path): return pl.DataFrame()

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"🎯 MEGA FILE smart processing {os.path.basename(file_path)} ({file_size_mb:.1f} MB)")

        if file_size_mb < 1000:
            print(f"   File too small for smart processing, using normal streaming...")
            return load_and_aggregate_streaming(file_path, ["user_id_hashed", "content_id_hashed"], agg_exprs)

        session_pairs = sessions_df.select(["user_id_hashed", "content_id_hashed"]).unique()
        session_pairs_set = set(
            (row["user_id_hashed"], row["content_id_hashed"]) for row in session_pairs.iter_rows(named=True))
        del session_pairs

        scan = pl.scan_parquet(file_path)
        total_rows = scan.select(pl.len()).collect().item()

        agg_dict = {}
        priority_kept = sample_kept = 0

        for start in range(0, total_rows, chunk_size):
            chunk = scan.slice(start, chunk_size).collect()
            if chunk.is_empty(): break

            processed_chunk = []
            for row in chunk.iter_rows(named=True):
                pair = (row["user_id_hashed"], row["content_id_hashed"])
                if pair in session_pairs_set:
                    processed_chunk.append(row)
                    priority_kept += 1
                elif np.random.random() < sample_rate:
                    processed_chunk.append(row)
                    sample_kept += 1

            if processed_chunk:
                processed_df = pl.DataFrame(processed_chunk)
                chunk_agg = processed_df.group_by(["user_id_hashed", "content_id_hashed"]).agg(agg_exprs)
                for row in chunk_agg.iter_rows(named=True):
                    key = (row["user_id_hashed"], row["content_id_hashed"])
                    if key not in agg_dict:
                        agg_dict[key] = {col: 0 for col in chunk_agg.columns if
                                         col not in ["user_id_hashed", "content_id_hashed"]}
                    for col in chunk_agg.columns:
                        if col not in ["user_id_hashed", "content_id_hashed"]:
                            agg_dict[key][col] += row[col] or 0
                del processed_df, chunk_agg

            del chunk
            if start % (chunk_size * 10) == 0: clean_memory()

        total_kept = priority_kept + sample_kept
        keep_rate = (total_kept / total_rows) * 100 if total_rows > 0 else 0
        print(
            f"   Final: {keep_rate:.1f}% kept ({total_kept:,}/{total_rows:,}) - Priority: {priority_kept:,}, Sampled: {sample_kept:,}")

        if not agg_dict: return pl.DataFrame()

        result_data = [{"user_id_hashed": key[0], "content_id_hashed": key[1], **values} for key, values in
                       agg_dict.items()]
        result = pl.DataFrame(result_data)
        print(f"✅ Fashion search smart processing - {len(result):,} pairs found")
        return result
    except Exception as e:
        print(f"❌ Error in smart fashion processing {os.path.basename(file_path)}: {e}")
        return pl.DataFrame()


def safe_join(main_df, join_df, on, how="left"):
    if join_df.is_empty(): return main_df
    print(f"   Joining {len(join_df):,} records...")
    result = main_df.join(join_df, on=on, how=how)
    del join_df
    clean_memory()
    return result


def compute_group_ndcg(y_true, y_score, groups, k=10):
    ndcgs = []
    ptr = 0
    for g in groups:
        g = int(g)
        if g <= 0: continue
        y_t = np.array(y_true[ptr:ptr + g])
        y_s = np.array(y_score[ptr:ptr + g])
        ptr += g
        if y_t.sum() == 0: continue
        try:
            nd = ndcg_score([y_t], [y_s], k=k)
            ndcgs.append(nd)
        except:
            continue
    return np.mean(ndcgs) if ndcgs else 0.0


def grid_search_lightgbm_ranker(train_X, train_y, train_groups, val_X, val_y, val_groups, param_grid,
                                objective='lambdarank', early_stopping_rounds=30):
    best_score = -1
    best_params = None

    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        print(f"Testing params: {params}")

        model = lgb.LGBMRanker(
            objective=objective,
            metric='ndcg',
            boosting_type='gbdt',
            n_estimators=int(params.get('n_estimators', 100)),
            num_leaves=int(params.get('num_leaves', 31)),
            learning_rate=float(params.get('learning_rate', 0.05)),
            feature_fraction=float(params.get('feature_fraction', 0.8)),
            bagging_fraction=float(params.get('bagging_fraction', 0.8)),
            bagging_freq=5,
            min_child_samples=int(params.get('min_child_samples', 20)),
            reg_lambda=float(params.get('reg_lambda', 0.0)),
            random_state=SEED,
            verbose=-1,
            n_jobs=-1
        )
        try:
            model.fit(train_X, train_y, group=train_groups,
                      eval_set=[(val_X, val_y)], eval_group=[val_groups],
                      callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                                 lgb.log_evaluation(period=0)])

            val_preds = model.predict(val_X, num_iteration=getattr(model, "best_iteration_", None) or None)
            ndcg_val = compute_group_ndcg(val_y, val_preds, val_groups, k=10)
            print(f"   -> ndcg@10: {ndcg_val:.5f}")

            if ndcg_val > best_score:
                best_score = ndcg_val
                best_params = params
        except Exception as e:
            print(f"   training failed for {params}: {e}")
            continue

    return best_params, best_score


def main():
    start = datetime.now()
    print("🚀 Starting improved LGBMRanker pipeline")

    # 1. Load sessions
    print("📋 Loading sessions...")
    train_sessions = pl.scan_parquet(f"{DATA_PATH}/train_sessions.parquet").collect()
    test_sessions = pl.scan_parquet(f"{DATA_PATH}/test_sessions.parquet").collect()
    all_sessions = pl.concat([train_sessions.select(["user_id_hashed", "content_id_hashed"]),
                              test_sessions.select(["user_id_hashed", "content_id_hashed"])])

    # Time features
    time_features = [pl.col("ts_hour").dt.hour().alias("hour_of_day"),
                     pl.col("ts_hour").dt.weekday().alias("day_of_week"),
                     pl.col("ts_hour").dt.date().alias("ts_date")]
    train_sessions = train_sessions.with_columns(time_features)
    test_sessions = test_sessions.with_columns(time_features)
    print(f"   Train: {len(train_sessions):,}, Test: {len(test_sessions):,}")

    # 2. Load metadata
    print("📦 Loading metadata...")
    content_meta = pl.scan_parquet(f"{DATA_PATH}/content/metadata.parquet").select([
        "content_id_hashed", "level1_category_name", "level2_category_name", "leaf_category_name",
        "attribute_type_count", "total_attribute_option_count", "merchant_count", "filterable_label_count",
        "content_creation_date", "cv_tags"]).collect()
    train_sessions = safe_join(train_sessions, content_meta, on="content_id_hashed")
    test_sessions = safe_join(test_sessions, content_meta, on="content_id_hashed")
    del content_meta

    # 3. Price data
    print("💰 Loading price data...")
    price_data = pl.scan_parquet(f"{DATA_PATH}/content/price_rate_review_data.parquet").select([
        "content_id_hashed", "update_date", "original_price", "selling_price", "discounted_price",
        "content_review_count", "content_review_wth_media_count", "content_rate_count", "content_rate_avg"]).collect()
    price_data = price_data.with_columns(pl.col("update_date").dt.date().alias("ts_date"))
    train_sessions = safe_join(train_sessions, price_data, on=["content_id_hashed", "ts_date"])
    test_sessions = safe_join(test_sessions, price_data, on=["content_id_hashed", "ts_date"])
    del price_data

    # 4. Content signals
    print("📊 Processing content signals...")
    files_and_configs = [
        (f"{DATA_PATH}/content/search_log.parquet", ["content_id_hashed"],
         [pl.sum("total_search_impression").alias("content_impr"),
          pl.sum("total_search_click").alias("content_click")]),
        (f"{DATA_PATH}/content/sitewide_log.parquet", ["content_id_hashed"],
         [pl.sum("total_click").alias("content_total_click"), pl.sum("total_order").alias("content_total_order")]),
        (f"{DATA_PATH}/content/top_terms_log.parquet", ["content_id_hashed", "search_term_normalized"],
         [pl.sum("total_search_impression").alias("term_content_impr"),
          pl.sum("total_search_click").alias("term_content_click")])
    ]

    for file_path, group_cols, agg_exprs in files_and_configs:
        df = load_and_aggregate_streaming(file_path, group_cols, agg_exprs)
        if not df.is_empty():
            if "content_impr" in df.columns:
                df = df.with_columns((pl.col("content_click") / (pl.col("content_impr") + 1e-6)).alias("content_ctr"))
            elif "content_total_click" in df.columns:
                df = df.with_columns(
                    (pl.col("content_total_order") / (pl.col("content_total_click") + 1e-6)).alias("content_cvr"))
            elif "term_content_impr" in df.columns:
                df = df.with_columns(
                    (pl.col("term_content_click") / (pl.col("term_content_impr") + 1e-6)).alias("term_content_ctr"))

            train_sessions = safe_join(train_sessions, df, on=group_cols)
            test_sessions = safe_join(test_sessions, df, on=group_cols)

    # 5. User signals
    print("👤 Processing user signals...")
    user_meta = pl.scan_parquet(f"{DATA_PATH}/user/metadata.parquet").collect()
    user_meta = user_meta.with_columns((2025 - pl.col("user_birth_year")).alias("user_age"))
    user_meta = user_meta.with_columns([
        (pl.col("user_age") < 25).cast(pl.Int8).alias("user_is_young"),
        ((pl.col("user_age") >= 25) & (pl.col("user_age") <= 45)).cast(pl.Int8).alias("user_is_adult"),
        (pl.col("user_age") > 45).cast(pl.Int8).alias("user_is_senior")
    ])

    if "user_tenure_in_days" in user_meta.columns:
        user_meta = user_meta.with_columns([
            (pl.col("user_tenure_in_days") < 180).cast(pl.Int8).alias("user_new"),
            ((pl.col("user_tenure_in_days") >= 180) & (pl.col("user_tenure_in_days") < 720)).cast(pl.Int8).alias(
                "user_mid"),
            (pl.col("user_tenure_in_days") >= 720).cast(pl.Int8).alias("user_old")
        ])

    if "user_gender" in user_meta.columns:
        user_meta = user_meta.with_columns([
            (pl.col("user_gender") == "female").cast(pl.Int8).alias("user_is_female"),
            (pl.col("user_gender") == "male").cast(pl.Int8).alias("user_is_male")
        ])

    train_sessions = safe_join(train_sessions, user_meta, on="user_id_hashed")
    test_sessions = safe_join(test_sessions, user_meta, on="user_id_hashed")
    del user_meta

    # User aggregations
    user_files_and_configs = [
        (f"{DATA_PATH}/user/sitewide_log.parquet", ["user_id_hashed"],
         [pl.sum("total_click").alias("user_total_click"), pl.sum("total_order").alias("user_total_order")]),
        (f"{DATA_PATH}/user/search_log.parquet", ["user_id_hashed"],
         [pl.sum("total_search_impression").alias("user_search_impr"),
          pl.sum("total_search_click").alias("user_search_click")]),
        (f"{DATA_PATH}/user/top_terms_log.parquet", ["user_id_hashed", "search_term_normalized"],
         [pl.sum("total_search_impression").alias("user_term_impr"),
          pl.sum("total_search_click").alias("user_term_click")]),
        (f"{DATA_PATH}/term/search_log.parquet", ["search_term_normalized"],
         [pl.sum("total_search_impression").alias("term_impr"), pl.sum("total_search_click").alias("term_click")])
    ]

    for file_path, group_cols, agg_exprs in user_files_and_configs:
        df = load_and_aggregate_streaming(file_path, group_cols, agg_exprs)
        if not df.is_empty():
            if "user_total_click" in df.columns:
                df = df.with_columns(
                    (pl.col("user_total_order") / (pl.col("user_total_click") + 1e-6)).alias("user_cvr"))
            elif "user_search_impr" in df.columns:
                df = df.with_columns([
                    (pl.col("user_search_click") / (pl.col("user_search_impr") + 1e-6)).alias("user_search_ctr"),
                    (pl.col("user_search_impr") > 0).cast(pl.Int8).alias("has_user_search_history")
                ])
            elif "user_term_impr" in df.columns:
                df = df.with_columns(
                    (pl.col("user_term_click") / (pl.col("user_term_impr") + 1e-6)).alias("user_term_ctr"))
            elif "term_impr" in df.columns:
                df = df.with_columns((pl.col("term_click") / (pl.col("term_impr") + 1e-6)).alias("term_ctr"))

            train_sessions = safe_join(train_sessions, df, on=group_cols)
            test_sessions = safe_join(test_sessions, df, on=group_cols)

    # 6. Large files processing
    print("🎯 Processing large fashion/user-content files...")
    fashion_search = smart_process_fashion_search(f"{DATA_PATH}/user/fashion_search_log.parquet", all_sessions,
                                                  [pl.sum("total_search_impression").alias("user_content_impr"),
                                                   pl.sum("total_search_click").alias("user_content_click")],
                                                  sample_rate=0.2)

    # 1. Fashion search dosyası işleniyor (büyük dosya, samplingli)
    if not fashion_search.is_empty():
        fashion_search = fashion_search.with_columns([
            (pl.col("user_content_click") / (pl.col("user_content_impr") + 1e-6)).alias("user_content_ctr"),
            (pl.col("user_content_impr") > 0).cast(pl.Int8).alias("has_user_content_history")
        ])
        train_sessions = safe_join(train_sessions, fashion_search, on=["user_id_hashed", "content_id_hashed"])
        test_sessions = safe_join(test_sessions, fashion_search, on=["user_id_hashed", "content_id_hashed"])

    # 2. Fashion sitewide dosyası işleniyor (tamamı RAM'e alınır)
    agg_exprs = [
        pl.sum("total_click").alias("user_content_total_click"),
        pl.sum("total_order").alias("user_content_total_order"),
        pl.sum("total_cart").alias("user_content_total_cart"),
        pl.sum("total_fav").alias("user_content_total_fav"),
    ]
    fashion_sitewide = load_and_aggregate_streaming(
        f"{DATA_PATH}/user/fashion_sitewide_log.parquet",
        ["user_id_hashed", "content_id_hashed"],
        agg_exprs
    )

    if not fashion_sitewide.is_empty():
        # Rename columns
        rename_map = {}
        for c in fashion_sitewide.columns:
            if 'click' in c.lower():
                rename_map[c] = 'user_content_total_click'
            elif 'order' in c.lower():
                rename_map[c] = 'user_content_total_order'
            elif 'cart' in c.lower():
                rename_map[c] = 'user_content_total_cart'
            elif 'fav' in c.lower():
                rename_map[c] = 'user_content_total_fav'
        if rename_map: fashion_sitewide = fashion_sitewide.rename(rename_map)

        # Derived features
        derived_exprs = []
        cols = set(fashion_sitewide.columns)
        if {'user_content_total_click', 'user_content_total_order'}.issubset(cols):
            derived_exprs.append(
                (pl.col('user_content_total_order') / (pl.col('user_content_total_click') + 1e-6)).alias(
                    'user_content_cvr'))
        if {'user_content_total_cart', 'user_content_total_click'}.issubset(cols):
            derived_exprs.append(
                (pl.col('user_content_total_cart') / (pl.col('user_content_total_click') + 1e-6)).alias(
                    'user_content_cart_ratio'))
        if {'user_content_total_fav', 'user_content_total_click'}.issubset(cols):
            derived_exprs.append((pl.col('user_content_total_fav') / (pl.col('user_content_total_click') + 1e-6)).alias(
                'user_content_fav_ratio'))
        if 'user_content_total_click' in cols:
            derived_exprs.append(
                (pl.col('user_content_total_click') > 0).cast(pl.Int8).alias('has_user_content_interaction'))

        if derived_exprs: fashion_sitewide = fashion_sitewide.with_columns(derived_exprs)
        train_sessions = safe_join(train_sessions, fashion_sitewide, on=["user_id_hashed", "content_id_hashed"])
        test_sessions = safe_join(test_sessions, fashion_sitewide, on=["user_id_hashed", "content_id_hashed"])

    del all_sessions
    clean_memory()

    # 7. Feature engineering
    print("⚙️ Feature engineering...")
    feature_exprs = [
        (pl.col("selling_price") / (pl.col("content_rate_avg") + 1e-6)).alias("price_to_rating"),
        (pl.col("content_review_count") > 0).cast(pl.Int8).alias("has_reviews"),
        (pl.col("content_rate_avg") > 4.0).cast(pl.Int8).alias("high_rating"),
        (pl.col("merchant_count") > 1).cast(pl.Int8).alias("multi_merchant"),
        (pl.col("ts_date") - pl.col("content_creation_date").dt.date()).dt.total_days().alias("content_freshness_days"),
        (pl.concat_str([pl.lit(" "), pl.col("cv_tags"), pl.lit(" ")]).str.contains(
            pl.concat_str([pl.lit(" "), pl.col("search_term_normalized"), pl.lit(" ")]))).cast(pl.Int8).alias(
            "term_exact_word_match"),
        (pl.col("cv_tags").str.contains(pl.col("search_term_normalized"))).cast(pl.Int8).alias("term_in_cv_tags"),
        ((pl.col("original_price") - pl.col("selling_price")) / (pl.col("original_price") + 1e-6)).alias(
            "discount_pct"),
        (pl.col("content_review_wth_media_count") / (pl.col("content_review_count") + 1e-6)).alias(
            "media_review_ratio"),
        ((pl.col('selling_price') - pl.col('selling_price').mean().over('session_id')) / (
                    pl.col('selling_price').std().over('session_id') + 1e-6)).alias('price_zscore'),
        ((pl.col('content_rate_avg') - pl.col('content_rate_avg').mean().over('session_id')) / (
                    pl.col('content_rate_avg').std().over('session_id') + 1e-6)).alias('rating_zscore')
    ]

    train_sessions = train_sessions.with_columns(feature_exprs)
    test_sessions = test_sessions.with_columns(feature_exprs)

    # Build session last-timestamp map for time-based split (A plan)
    print("📆 Building session last-timestamp map for time-based split...")
    try:
        session_ts_df = train_sessions.select(['session_id', 'ts_hour']).to_pandas()
        session_last_ts = session_ts_df.groupby('session_id')['ts_hour'].max()
        del session_ts_df
    except Exception as _e:
        print("   Failed to build session timestamp map:", _e)
        session_last_ts = None

    # Additional features
    if 'content_freshness_days' in train_sessions.columns:
        train_sessions = train_sessions.with_columns(
            (pl.col('content_freshness_days') <= 30).cast(pl.Int8).alias('is_new_product'))
        test_sessions = test_sessions.with_columns(
            (pl.col('content_freshness_days') <= 30).cast(pl.Int8).alias('is_new_product'))

    # 8. Handle missing values
    print("🔧 Handling missing values...")
    numeric_cols = ["selling_price", "original_price", "content_review_count", "content_rate_avg",
                    "attribute_type_count",
                    "merchant_count", "price_to_rating", "user_age", "content_freshness_days", "price_zscore",
                    "rating_zscore", "discount_pct"]

    current_cols = set(train_sessions.columns)
    for col in ['content_ctr', 'content_cvr', 'user_cvr', 'user_content_ctr', 'user_content_cvr', 'term_content_ctr']:
        if col in current_cols: numeric_cols.append(col)

    for col in numeric_cols:
        if col in train_sessions.columns:
            try:
                median_val = train_sessions.select(pl.col(col).median()).item()
                if median_val is not None:
                    train_sessions = train_sessions.with_columns(pl.col(col).fill_null(median_val))
                    test_sessions = test_sessions.with_columns(pl.col(col).fill_null(median_val))
            except:
                continue

    train_sessions = train_sessions.fill_null(0)
    test_sessions = test_sessions.fill_null(0)

    # 9. Prepare features for LGBMRanker
    print("🤖 Preparing LGBMRanker data...")
    feature_cols = ["selling_price", "original_price", "content_review_count", "content_rate_avg",
                    "attribute_type_count",
                    "merchant_count", "price_to_rating", "has_reviews", "high_rating", "multi_merchant", "user_age",
                    "hour_of_day", "day_of_week", "price_zscore", "rating_zscore", "content_freshness_days",
                    "is_new_product",
                    "term_exact_word_match", "term_in_cv_tags", "discount_pct", "media_review_ratio",
                    "user_is_young", "user_new", "user_old",
                    "user_is_female", "user_is_male"]

    # Add conditional features
    conditional_features = ["content_ctr", "content_cvr", "user_cvr", "user_content_ctr", "user_content_cvr",
                            "has_user_content_history", "has_user_content_interaction", "user_content_total_click",
                            "user_content_cart_ratio", "user_content_fav_ratio", "term_content_ctr", "user_search_ctr"]

    current_cols = set(train_sessions.columns)
    feature_cols.extend([f for f in conditional_features if f in current_cols])
    print(f"📊 Using {len(feature_cols)} features for LGBMRanker")

    # Convert to pandas with additional target variables
    train_X = train_sessions.select(feature_cols).to_pandas()
    train_y_click = train_sessions.select("clicked").to_pandas()["clicked"]
    train_y_order = train_sessions.select("ordered").to_pandas()["ordered"]
    train_y_cart = train_sessions.select("added_to_cart").to_pandas()["added_to_cart"]
    train_y_fav = train_sessions.select("added_to_fav").to_pandas()["added_to_fav"]
    train_sessions_ids = train_sessions.select("session_id").to_pandas()["session_id"]

    test_X = test_sessions.select(feature_cols).to_pandas()
    test_sessions_ids = test_sessions.select("session_id").to_pandas()["session_id"]
    test_content_ids = test_sessions.select("content_id_hashed").to_pandas()["content_id_hashed"]

    # free up large polars objects
    del train_sessions, test_sessions
    clean_memory()

    # 10. Prepare validation split (TIME-BASED)
    print("📋 Preparing time-based group data and local validation split...")
    import pandas as pd

    # helper: compute group sizes preserving group order from rows
    def groups_from_session_ids(series_like):
        s = pd.Series(series_like)
        uniq = s.drop_duplicates()
        sizes = s.groupby(s).size().loc[uniq.values].values
        return sizes

    if session_last_ts is None or len(session_last_ts) < 10:
        # fallback to random split if timestamp map not available
        print("   session_last_ts not found or too small — falling back to random split")
        unique_sessions = np.array(list(train_sessions_ids.unique()))
        np.random.seed(SEED)
        np.random.shuffle(unique_sessions)
        val_session_ids = set(unique_sessions[:int(len(unique_sessions) * 0.01)])
    else:
        sorted_sessions = session_last_ts.sort_values()
        n_val = max(1, int(len(sorted_sessions) * 0.05))
        val_session_ids = set(sorted_sessions.index[-n_val:])  # latest 10% of sessions
        print(f"   Using latest {n_val} sessions as validation (time-based)")

    train_mask = train_sessions_ids.apply(lambda x: x not in val_session_ids)
    val_mask = ~train_mask

    X_tr, y_tr_click = train_X[train_mask], train_y_click[train_mask]
    X_val, y_val_click = train_X[val_mask], train_y_click[val_mask]

    y_tr_order = train_y_order[train_mask]
    y_val_order = train_y_order[val_mask]

    y_tr_cart = train_y_cart[train_mask]
    y_val_cart = train_y_cart[val_mask]

    y_tr_fav = train_y_fav[train_mask]
    y_val_fav = train_y_fav[val_mask]

    # group arrays in the order rows appear
    train_groups = groups_from_session_ids(train_sessions_ids[train_mask].values)
    val_groups = groups_from_session_ids(train_sessions_ids[val_mask].values)

    # cleanup large objects
    try:
        del session_last_ts
    except:
        pass

    # 11. Grid search for click
    print("🔎 Grid search for light tuning (click model)...")
    param_grid_click = {
        'num_leaves': [31, 63],
        'learning_rate': [0.01, 0.03],
        'n_estimators': [200, 300],
        'feature_fraction': [0.8, 0.9],
        'bagging_fraction': [0.7, 0.8],
        'min_child_samples': [20, 40],
        'reg_lambda': [1.0]
    }
    best_params_click, best_score_click = grid_search_lightgbm_ranker(X_tr, y_tr_click, train_groups, X_val,
                                                                      y_val_click, val_groups, param_grid_click)
    print(f"Best click params: {best_params_click}, ndcg@10: {best_score_click}")

    # Grid search for order
    print("🔎 Grid search for light tuning (order model)...")
    param_grid_order = {
        'num_leaves': [31, 63],
        'learning_rate': [0.01, 0.03],
        'n_estimators': [200, 300],
        'feature_fraction': [0.8, 0.9],
        'bagging_fraction': [0.7, 0.8],
        'min_child_samples': [20, 40],
        'reg_lambda': [1.0]
    }
    best_params_order, best_score_order = grid_search_lightgbm_ranker(X_tr, y_tr_order, train_groups, X_val,
                                                                      y_val_order, val_groups, param_grid_order)
    print(f"Best order params: {best_params_order}, ndcg@10: {best_score_order}")

    # 12. Train multi-objective ensemble
    print("🏆 Training enhanced multi-objective ensemble...")
    click_rankers, order_rankers, cart_rankers, fav_rankers = [], [], [], []

    # Base parameters for each objective
    p_click = best_params_click or {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100,
                                    'feature_fraction': 0.8, 'bagging_fraction': 0.8}
    p_order = best_params_order or {'num_leaves': 63, 'learning_rate': 0.03, 'n_estimators': 150,
                                    'feature_fraction': 0.7, 'bagging_fraction': 0.7}
    p_cart = {'num_leaves': 47, 'learning_rate': 0.04, 'n_estimators': 125, 'feature_fraction': 0.75,
              'bagging_fraction': 0.75}
    p_fav = {'num_leaves': 47, 'learning_rate': 0.04, 'n_estimators': 125, 'feature_fraction': 0.75,
             'bagging_fraction': 0.75}

    for seed in [42, 43, 44, 45, 46]:
        # Click model
        click_ranker = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='gbdt',
                                      num_leaves=int(p_click['num_leaves']),
                                      learning_rate=float(p_click['learning_rate']),
                                      feature_fraction=float(p_click.get('feature_fraction', 0.8)),
                                      bagging_fraction=float(p_click.get('bagging_fraction', 0.8)),
                                      bagging_freq=5, min_child_samples=20, n_estimators=int(p_click['n_estimators']),
                                      random_state=seed, verbose=-1)
        click_ranker.fit(X_tr, y_tr_click, group=train_groups)
        click_rankers.append(click_ranker)

        # Order model
        order_ranker = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='gbdt',
                                      num_leaves=int(p_order['num_leaves']),
                                      learning_rate=float(p_order['learning_rate']),
                                      feature_fraction=float(p_order.get('feature_fraction', 0.7)),
                                      bagging_fraction=float(p_order.get('bagging_fraction', 0.7)),
                                      bagging_freq=5, min_child_samples=20, n_estimators=int(p_order['n_estimators']),
                                      random_state=seed, verbose=-1)
        order_ranker.fit(X_tr, y_tr_order, group=train_groups)
        order_rankers.append(order_ranker)

        # Cart model
        cart_ranker = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='gbdt',
                                     num_leaves=int(p_cart['num_leaves']), learning_rate=float(p_cart['learning_rate']),
                                     feature_fraction=float(p_cart.get('feature_fraction', 0.75)),
                                     bagging_fraction=float(p_cart.get('bagging_fraction', 0.75)),
                                     bagging_freq=5, min_child_samples=20, n_estimators=int(p_cart['n_estimators']),
                                     random_state=seed, verbose=-1)
        cart_ranker.fit(X_tr, y_tr_cart, group=train_groups)
        cart_rankers.append(cart_ranker)

        # Favorite model
        fav_ranker = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='gbdt',
                                    num_leaves=int(p_fav['num_leaves']), learning_rate=float(p_fav['learning_rate']),
                                    feature_fraction=float(p_fav.get('feature_fraction', 0.75)),
                                    bagging_fraction=float(p_fav.get('bagging_fraction', 0.75)),
                                    bagging_freq=5, min_child_samples=20, n_estimators=int(p_fav['n_estimators']),
                                    random_state=seed, verbose=-1)
        fav_ranker.fit(X_tr, y_tr_fav, group=train_groups)
        fav_rankers.append(fav_ranker)

    clean_memory()

    # 13. Feature importance
    print("📈 Feature importances (click model ensemble average):")
    importances_click = np.mean([r.feature_importances_ for r in click_rankers], axis=0)
    for feat, imp in sorted(zip(feature_cols, importances_click), key=lambda x: -x[1])[:50]:
        print(f"   {feat}: {imp}")

    print("📈 Feature importances (order model ensemble average):")
    importances_order = np.mean([r.feature_importances_ for r in order_rankers], axis=0)
    for feat, imp in sorted(zip(feature_cols, importances_order), key=lambda x: -x[1])[:50]:
        print(f"   {feat}: {imp}")

    # 14. Diagnostics
    print("🔎 Diagnostics: computing train/val session AUC, ndcg and feature correlations...")

    def mean_session_auc(session_ids, y, preds):
        import pandas as _pd
        _df = _pd.DataFrame({"session_id": session_ids, "y": y, "pred": preds})
        _aucs = []
        for s, g in _df.groupby("session_id"):
            if g["y"].sum() == 0 or g["y"].nunique() == 1: continue
            try:
                _aucs.append(roc_auc_score(g["y"], g["pred"]))
            except:
                continue
        return (np.mean(_aucs) if _aucs else 0.0), len(_aucs)

    try:
        # For click
        train_preds_click = np.mean([r.predict(X_tr) for r in click_rankers], axis=0)
        val_preds_click = np.mean([r.predict(X_val) for r in click_rankers], axis=0)

        train_auc_click, n_train_auc_click = mean_session_auc(train_sessions_ids[train_mask].values, y_tr_click.values,
                                                              train_preds_click)
        val_auc_click, n_val_auc_click = mean_session_auc(train_sessions_ids[val_mask].values, y_val_click.values,
                                                          val_preds_click)
        print(f"   [Click] Train mean session AUC: {train_auc_click:.5f} (n_sessions={n_train_auc_click})")
        print(f"   [Click] Val   mean session AUC: {val_auc_click:.5f} (n_sessions={n_val_auc_click})")

        ndcg_val_click = compute_group_ndcg(y_val_click.values, val_preds_click, val_groups, k=10)
        print(f"   [Click] Val ndcg@10 (computed): {ndcg_val_click:.5f}")

        # For order
        train_preds_order = np.mean([r.predict(X_tr) for r in order_rankers], axis=0)
        val_preds_order = np.mean([r.predict(X_val) for r in order_rankers], axis=0)

        train_auc_order, n_train_auc_order = mean_session_auc(train_sessions_ids[train_mask].values, y_tr_order.values,
                                                              train_preds_order)
        val_auc_order, n_val_auc_order = mean_session_auc(train_sessions_ids[val_mask].values, y_val_order.values,
                                                          val_preds_order)
        print(f"   [Order] Train mean session AUC: {train_auc_order:.5f} (n_sessions={n_train_auc_order})")
        print(f"   [Order] Val   mean session AUC: {val_auc_order:.5f} (n_sessions={n_val_auc_order})")

        ndcg_val_order = compute_group_ndcg(y_val_order.values, val_preds_order, val_groups, k=10)
        print(f"   [Order] Val ndcg@10 (computed): {ndcg_val_order:.5f}")
    except Exception as _e:
        print("   Diagnostics train/val AUC failed:", _e)

    # Feature correlations
    try:
        import pandas as _pd, scipy.stats as _st
        _corrs_click = []
        for f in X_tr.columns:
            try:
                c = _st.spearmanr(X_tr[f].fillna(0).values, y_tr_click.values).correlation
                _corrs_click.append((f, abs(c if c is not None else 0)))
            except:
                _corrs_click.append((f, 0.0))
        _corrs_sorted_click = sorted(_corrs_click, key=lambda x: -x[1])[:20]
        print("   Top feature correlations (abs Spearman) vs clicked:")
        for f, c in _corrs_sorted_click:
            print(f"     {f}: {c:.4f}")

        _corrs_order = []
        for f in X_tr.columns:
            try:
                c = _st.spearmanr(X_tr[f].fillna(0).values, y_tr_order.values).correlation
                _corrs_order.append((f, abs(c if c is not None else 0)))
            except:
                _corrs_order.append((f, 0.0))
        _corrs_sorted_order = sorted(_corrs_order, key=lambda x: -x[1])[:20]
        print("   Top feature correlations (abs Spearman) vs ordered:")
        for f, c in _corrs_sorted_order:
            print(f"     {f}: {c:.4f}")
    except Exception as _e:
        print("   Feature correlation failed:", _e)

    print("🔎 Diagnostics complete. Proceeding to predictions...")

    # 15. Predictions and submission with enhanced scoring
    print("🔮 Generating multi-objective ensemble predictions...")
    click_scores = np.mean([r.predict(test_X) for r in click_rankers], axis=0)
    order_scores = np.mean([r.predict(test_X) for r in order_rankers], axis=0)
    cart_scores = np.mean([r.predict(test_X) for r in cart_rankers], axis=0)
    fav_scores = np.mean([r.predict(test_X) for r in fav_rankers], axis=0)

    del test_X, click_rankers, order_rankers, cart_rankers, fav_rankers
    clean_memory()

    print("📤 Creating enhanced submission...")
    submission_df = pl.DataFrame({
        "session_id": test_sessions_ids,
        "content_id_hashed": test_content_ids,
        "click_score": click_scores,
        "order_score": order_scores,
        "cart_score": cart_scores,
        "fav_score": fav_scores
    })

    # Enhanced scoring with cart and favorite signals
    submission_df = submission_df.with_columns(
        (0.25 * pl.col("click_score") +
         0.45 * pl.col("order_score") +
         0.20 * pl.col("cart_score") +
         0.10 * pl.col("fav_score")).alias("final_score"))
    submission_df = submission_df.sort(["session_id", "final_score"], descending=[False, True])

    submission = (submission_df.group_by("session_id").agg(pl.col("content_id_hashed").alias("prediction"))
                  .with_columns(pl.col("prediction").list.join(" ").alias("prediction"))
                  .select(["session_id", "prediction"]))

    submission.write_csv("submission14.csv")

    total_time = datetime.now() - start
    print(f"✅ Pipeline completed in {total_time.total_seconds():.0f}s")
    print(f"📄 Submission: submission14.csv")


if __name__ == "__main__":
    main()
