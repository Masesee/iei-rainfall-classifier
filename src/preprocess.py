import re
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from text_unidecode import unidecode


PUNCTUATION_REGEX = re.compile(r"[\.!?,;:\-\(\)\[\]\{\}/\\'\"`~@#$%^&*_+=|<>]+")
WHITESPACE_REGEX = re.compile(r"\s+")


def _clean_text(value: str, fillna_token: str = "<missing>") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fillna_token
    if not isinstance(value, str):
        value = str(value)
    value = unidecode(value)
    value = value.lower()
    value = PUNCTUATION_REGEX.sub(" ", value)
    value = WHITESPACE_REGEX.sub(" ", value).strip()
    return value if value else fillna_token


def standardize_datetimes(df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def enforce_dtypes(df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]) -> pd.DataFrame:
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def map_target_labels(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if target_col not in df.columns:
        return df, {}
    unique_labels = [lbl for lbl in df[target_col].dropna().unique()]
    # Stable sort for deterministic mapping
    unique_labels = sorted(unique_labels)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    df[target_col + "_label"] = df[target_col].map(label_to_id)
    return df, label_to_id


def load_and_clean_data(path: str,
                        config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    cfg = config or {}
    columns_cfg = cfg.get("columns", {})
    preprocessing_cfg = cfg.get("preprocessing", {})

    id_col = columns_cfg.get("id", "ID")
    target_col = columns_cfg.get("target", "Target")
    datetime_cols = columns_cfg.get("datetime_columns", [])
    text_cols = columns_cfg.get("text_columns", [])
    categorical_cols = columns_cfg.get("categorical_columns", [])
    numerical_cols = columns_cfg.get("numerical_columns", [])

    text_clean_cfg = preprocessing_cfg.get("text_clean", {})
    fillna_token = text_clean_cfg.get("fillna_token", "<missing>")

    df = pd.read_csv(path)

    # Basic cleaning
    # Strip spaces in column names
    df.columns = [c.strip() for c in df.columns]

    # Standardize datetimes
    df = standardize_datetimes(df, datetime_cols)

    # Text cleaning
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _clean_text(x, fillna_token=fillna_token))

    # Handle ID presence
    if id_col not in df.columns:
        # Create a synthetic ID if not present
        df[id_col] = np.arange(len(df))

    # Enforce types
    df = enforce_dtypes(df, categorical_cols, numerical_cols)

    # Target mapping (only if target present)
    df, label_to_id = map_target_labels(df, target_col)

    return df, label_to_id


def save_label_mapping(mapping: Dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_label_mapping(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
