import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from scipy import sparse
import lightgbm as lgb

from preprocess import save_label_mapping
from features import create_features, transform_tfidf


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_feature_matrix(df_feat: pd.DataFrame,
                          feature_cols,
                          tfidf_matrix) -> sparse.csr_matrix:
    X_dense = df_feat[feature_cols].astype(float).fillna(0.0).values
    X_dense_sparse = sparse.csr_matrix(X_dense)
    if tfidf_matrix is not None:
        X = sparse.hstack([X_dense_sparse, tfidf_matrix], format="csr")
    else:
        X = X_dense_sparse
    return X


def train_model(train_df: pd.DataFrame,
                config: Dict) -> Tuple[float, pd.DataFrame]:
    paths = config.get("paths", {})
    columns_cfg = config.get("columns", {})
    training_cfg = config.get("training", {})
    features_cfg = config.get("features", {})

    model_dir = paths.get("model_output_dir", "models/baseline_lgbm")
    encoders_dir = paths.get("encoders_dir", os.path.join(model_dir, "encoders"))
    vectorizer_prefix = paths.get("vectorizer_prefix")
    oof_path = paths.get("oof_predictions_path", os.path.join(model_dir, "oof_predictions.csv"))
    label_map_path = paths.get("label_mapping_path", os.path.join(model_dir, "label_mapping.json"))
    feature_columns_path = paths.get("feature_columns_path", os.path.join(model_dir, "feature_columns.json"))

    _ensure_dir(model_dir)
    _ensure_dir(encoders_dir)

    id_col = columns_cfg.get("id", "ID")
    target_col = columns_cfg.get("target", "Target")
    group_col = columns_cfg.get("group", "user_id")

    # Target numeric
    target_numeric_col = f"{target_col}_label"
    if target_numeric_col not in train_df.columns:
        raise ValueError(f"Expected numeric target column '{target_numeric_col}' in training data.")

    y = train_df[target_numeric_col].astype(int).values
    groups = train_df[group_col].astype(str).values if group_col in train_df.columns else None

    # CV strategy: StratifiedGroupKFold
    n_splits = training_cfg.get("n_splits", 5)
    random_seed = training_cfg.get("random_seed", 42)
    shuffle = training_cfg.get("shuffle", True)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)

    oof_preds = np.zeros((len(train_df), len(np.unique(y))), dtype=float)
    fold_scores = []

    # Save label mapping size as LightGBM num_class
    num_classes = int(len(np.unique(y)))

    saved_feature_columns = False

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        # Feature engineering per fold
        tr_feat_df, tr_artifacts, tr_feature_cols = create_features(tr_df, config=config, context_df=tr_df, is_train=True, fold=fold)
        va_feat_df, va_artifacts, va_feature_cols = create_features(va_df, config=config, context_df=tr_df, is_train=True, fold=fold)

        # Align features
        feature_cols = tr_feature_cols

        # Save feature columns once
        if not saved_feature_columns:
            with open(feature_columns_path, "w", encoding="utf-8") as f:
                json.dump({"feature_columns": feature_cols}, f, indent=2)
            saved_feature_columns = True

        # TF-IDF matrices
        tfidf_prefix = paths.get("vectorizer_prefix")
        tfidf_matrix_tr = None
        tfidf_matrix_va = None
        tfidf_cfg = features_cfg.get("tfidf", {})
        tfidf_text_columns = tfidf_cfg.get("use_text_columns", [])
        if len(tfidf_text_columns) > 0 and tfidf_prefix:
            suffix = f"_{fold}"
            vectorizer_path = f"{tfidf_prefix}{suffix}.pkl"
            from_col = tfidf_text_columns[0]
            # Vectorizer already fitted in create_features for train; reload and transform
            tfidf_matrix_tr = transform_tfidf(tr_feat_df[from_col], vectorizer_path)
            tfidf_matrix_va = transform_tfidf(va_feat_df[from_col], vectorizer_path)

        # Build matrices
        X_tr = _build_feature_matrix(tr_feat_df, feature_cols, tfidf_matrix_tr)
        X_va = _build_feature_matrix(va_feat_df, feature_cols, tfidf_matrix_va)
        y_tr = tr_df[target_numeric_col].astype(int).values
        y_va = va_df[target_numeric_col].astype(int).values

        # Model
        lgb_params = training_cfg.get("lightgbm_params", {}).copy()
        lgb_params["num_class"] = num_classes
        clf = lgb.LGBMClassifier(**lgb_params)

        # Fit
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=training_cfg.get("early_stopping_rounds", 100), verbose=False),
                lgb.log_evaluation(period=training_cfg.get("verbose_eval", 100))
            ]
        )

        # Predict
        va_pred_proba = clf.predict_proba(X_va)
        oof_preds[va_idx] = va_pred_proba
        va_pred = np.argmax(va_pred_proba, axis=1)
        score = f1_score(y_va, va_pred, average="macro")
        fold_scores.append(score)
        print(f"Fold {fold} Macro F1: {score:.5f}")

        # Save model
        joblib.dump(clf, os.path.join(model_dir, f"model_fold_{fold}.pkl"))

        # Save TE states for this fold
        te_states = tr_artifacts.get("target_encoding_states", {})
        with open(os.path.join(encoders_dir, f"target_encoding_states_fold_{fold}.json"), "w", encoding="utf-8") as f:
            json.dump(te_states, f)

    # OOF score
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    oof_score = f1_score(y, oof_pred_labels, average="macro")
    print(f"OOF Macro F1: {oof_score:.5f}")

    # Save OOF
    oof_df = pd.DataFrame({
        columns_cfg.get("id", "ID"): train_df[id_col].values,
        f"{target_col}_label": y,
        "pred_label": oof_pred_labels,
    })
    oof_df.to_csv(oof_path, index=False)

    return oof_score, oof_df
