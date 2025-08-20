import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
import joblib


def add_temporal_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if datetime_col not in df.columns:
        return df
    series = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    df[f"{datetime_col}_hour"] = series.dt.hour
    df[f"{datetime_col}_dayofweek"] = series.dt.dayofweek
    df[f"{datetime_col}_month"] = series.dt.month
    return df


def add_user_submission_count(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    if user_col not in df.columns:
        return df
    counts = df[user_col].value_counts().to_dict()
    df[f"{user_col}_submission_count"] = df[user_col].map(counts).fillna(0).astype(int)
    return df


def fit_transform_tfidf(train_texts: pd.Series,
                        tfidf_cfg: Dict,
                        artifact_path_prefix: str,
                        fold: Optional[int] = None) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 2000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 3),
        max_df=tfidf_cfg.get("max_df", 0.95),
        stop_words=tfidf_cfg.get("stop_words", "english"),
    )
    X = vectorizer.fit_transform(train_texts.fillna(""))
    # Save per-fold or single
    suffix = f"_{fold}" if fold is not None else ""
    joblib.dump(vectorizer, f"{artifact_path_prefix}{suffix}.pkl")
    return vectorizer, X


def transform_tfidf(texts: pd.Series, vectorizer_path: str) -> np.ndarray:
    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    return vectorizer.transform(texts.fillna(""))


def safe_target_encoding(train_df: pd.DataFrame,
                         valid_df: pd.DataFrame,
                         column: str,
                         target_col: str,
                         smoothing: float = 10.0,
                         prior: Optional[float] = None) -> Tuple[pd.Series, pd.Series, Dict]:
    # Compute global prior
    if prior is None:
        prior = train_df[target_col].mean() if train_df[target_col].dtype != object else 0.0
    stats = train_df.groupby(column)[target_col].agg(["mean", "count"]) if column in train_df.columns else pd.DataFrame(columns=["mean","count"]).astype({"mean":float,"count":int})
    means = stats["mean"] if "mean" in stats else pd.Series(dtype=float)
    counts = stats["count"] if "count" in stats else pd.Series(dtype=int)
    smoothing_values = 1 / (1 + np.exp(-(counts - 1) / smoothing))
    encodings = prior * (1 - smoothing_values) + means * smoothing_values

    # Map
    train_encoded = train_df[column].map(encodings).fillna(prior)
    valid_encoded = valid_df[column].map(encodings).fillna(prior)

    return train_encoded, valid_encoded, {"prior": float(prior), "encodings": encodings.to_dict()}


def apply_target_encoding(test_df: pd.DataFrame, column: str, state: Dict) -> pd.Series:
    prior = state.get("prior", 0.0)
    encodings = state.get("encodings", {})
    return test_df[column].map(encodings).fillna(prior)


def create_features(df: pd.DataFrame,
                    config: Dict,
                    context_df: Optional[pd.DataFrame] = None,
                    is_train: bool = True,
                    fold: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    columns_cfg = config.get("columns", {})
    features_cfg = config.get("features", {})

    id_col = columns_cfg.get("id", "ID")
    target_col = columns_cfg.get("target", "Target_label")
    datetime_cols = columns_cfg.get("datetime_columns", [])
    categorical_cols = columns_cfg.get("categorical_columns", [])
    numerical_cols = columns_cfg.get("numerical_columns", [])
    text_cols = columns_cfg.get("text_columns", [])

    temporal_cfg = features_cfg.get("temporal", {})
    tfidf_cfg = features_cfg.get("tfidf", {})
    te_cfg = features_cfg.get("target_encoding", {})
    user_cfg = features_cfg.get("user", {})

    work_df = df.copy()

    # Temporal
    base_datetime = temporal_cfg.get("from_datetime")
    if base_datetime:
        work_df = add_temporal_features(work_df, base_datetime)

    # User submission count
    if user_cfg.get("submission_count", True):
        group_col = columns_cfg.get("group", "user_id")
        work_df = add_user_submission_count(work_df, group_col)

    # Target encoding across high-cardinality categorical columns
    te_states: Dict[str, Dict] = {}
    te_feature_names: List[str] = []
    target_numeric_col = f"{columns_cfg.get('target', 'Target')}_label"

    if is_train:
        # Target encoding using context_df for stability if provided
        ref_df = context_df if context_df is not None else work_df
        for col in categorical_cols:
            if col == target_numeric_col:
                continue
            tr_enc, val_enc, state = safe_target_encoding(ref_df, work_df, col, target_numeric_col, te_cfg.get("smoothing", 10.0))
            new_col = f"te_{col}"
            work_df[new_col] = val_enc.values
            te_states[col] = state
            te_feature_names.append(new_col)
    else:
        for col in categorical_cols:
            state = te_cfg.get("states", {}).get(col)
            if state is None:
                continue
            new_col = f"te_{col}"
            work_df[new_col] = apply_target_encoding(work_df, col, state)
            te_feature_names.append(new_col)

    # TF-IDF
    tfidf_text_columns = tfidf_cfg.get("use_text_columns", [])
    tfidf_prefix = config.get("paths", {}).get("vectorizer_prefix")

    tfidf_feature_names: List[str] = []
    if len(tfidf_text_columns) > 0 and tfidf_prefix:
        text_col = tfidf_text_columns[0]
        if is_train:
            vectorizer, X = fit_transform_tfidf(work_df[text_col], tfidf_cfg, tfidf_prefix, fold)
        else:
            suffix = f"_{fold}" if fold is not None else ""
            vectorizer_path = f"{tfidf_prefix}{suffix}.pkl"
            X = transform_tfidf(work_df[text_col], vectorizer_path)
        # Convert sparse matrix to DataFrame (dense can be big; keep sparse in training)
        tfidf_cols = [f"tfidf_{i}" for i in range(X.shape[1])]
        tfidf_feature_names = tfidf_cols
        # Keep as sparse placeholder; caller will horizontally stack during model training
        work_df["__tfidf_placeholder__"] = 0  # indicates presence
    else:
        X = None
        tfidf_feature_names = []

    # Collect final feature columns (excluding ID and target)
    base_feature_cols: List[str] = []
    # Numerical
    for col in numerical_cols:
        if col in work_df.columns:
            base_feature_cols.append(col)
    # Temporal columns added
    if base_datetime:
        base_feature_cols.extend([
            f"{base_datetime}_hour",
            f"{base_datetime}_dayofweek",
            f"{base_datetime}_month",
        ])
    # User submission count
    if user_cfg.get("submission_count", True):
        base_feature_cols.append(f"{columns_cfg.get('group', 'user_id')}_submission_count")
    # Target encodings
    base_feature_cols.extend(te_feature_names)

    artifact_state = {
        "target_encoding_states": te_states,
        "tfidf_feature_names": tfidf_feature_names,
    }

    return work_df, artifact_state, base_feature_cols
