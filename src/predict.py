import glob
import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from preprocess import load_and_clean_data, load_label_mapping
from features import create_features, transform_tfidf


def _build_feature_matrix(df_feat: pd.DataFrame,
                          feature_cols,
                          tfidf_matrix) -> sparse.csr_matrix:
    from scipy import sparse as _s
    X_dense = df_feat[feature_cols].astype(float).fillna(0.0).values
    X_dense_sparse = _s.csr_matrix(X_dense)
    if tfidf_matrix is not None:
        X = _s.hstack([X_dense_sparse, tfidf_matrix], format="csr")
    else:
        X = X_dense_sparse
    return X


def generate_predictions(test_df: pd.DataFrame, model_dir: str, config: Dict) -> pd.DataFrame:
    paths = config.get("paths", {})
    columns_cfg = config.get("columns", {})
    features_cfg = config.get("features", {})

    id_col = columns_cfg.get("id", "ID")
    target_col = columns_cfg.get("target", "Target")

    encoders_dir = paths.get("encoders_dir", os.path.join(model_dir, "encoders"))
    vectorizer_prefix = paths.get("vectorizer_prefix")

    # Infer number of folds by saved models
    model_paths = sorted(glob.glob(os.path.join(model_dir, "model_fold_*.pkl")))
    if len(model_paths) == 0:
        raise FileNotFoundError(f"No models found in {model_dir}")

    # Aggregate predictions
    fold_probs = []

    for model_path in model_paths:
        fold = int(os.path.basename(model_path).split("_")[-1].split(".")[0])

        # Load TE states for this fold into config for test-time application
        te_state_path = os.path.join(encoders_dir, f"target_encoding_states_fold_{fold}.json")
        with open(te_state_path, "r", encoding="utf-8") as f:
            te_states = json.load(f)
        # Inject states into config for use inside create_features
        config_feat = json.loads(json.dumps(config))
        config_feat["features"]["target_encoding"]["states"] = te_states

        # Feature engineering (test)
        test_feat_df, feat_artifacts, feature_cols = create_features(test_df, config=config_feat, is_train=False, fold=fold)

        # TF-IDF matrix
        tfidf_cfg = features_cfg.get("tfidf", {})
        tfidf_text_columns = tfidf_cfg.get("use_text_columns", [])
        if len(tfidf_text_columns) > 0 and vectorizer_prefix:
            suffix = f"_{fold}"
            vectorizer_path = f"{vectorizer_prefix}{suffix}.pkl"
            tfidf_matrix = transform_tfidf(test_feat_df[tfidf_text_columns[0]], vectorizer_path)
        else:
            tfidf_matrix = None

        X_test = _build_feature_matrix(test_feat_df, feature_cols, tfidf_matrix)

        # Load model and predict
        clf = joblib.load(model_path)
        proba = clf.predict_proba(X_test)
        fold_probs.append(proba)

    # Average probabilities
    mean_probs = np.mean(fold_probs, axis=0)
    pred_labels = np.argmax(mean_probs, axis=1)

    # Load label mapping to invert
    label_map_path = paths.get("label_mapping_path")
    if label_map_path and os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_to_id = json.load(f)
        id_to_label = {v: k for k, v in label_to_id.items()}
        pred_str = [id_to_label.get(int(i), str(i)) for i in pred_labels]
    else:
        pred_str = pred_labels.astype(str).tolist()

    submission = pd.DataFrame({
        id_col: test_df[id_col].values,
        target_col: pred_str,
    })

    sub_path = paths.get("submission_path", os.path.join("submissions", "submission.csv"))
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    submission.to_csv(sub_path, index=False)

    return submission
