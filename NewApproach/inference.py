#!/usr/bin/env python3
"""
inference.py
Run inference on windowed BGL data using saved classification and regression pipelines.

Usage:
    python inference.py \
      --windows data/bgl_windows_v1.parquet \
      --models  models/ \
      --threshold 0.5

Outputs:
    predictions.csv with columns: window_start, node_id, p_fail, eta_min, alert
"""
import argparse
import joblib
import pandas as pd
from pathlib import Path


def align_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Return df with exactly the given columns, missing filled with 0, extra dropped.
    """
    return df.reindex(columns=columns, fill_value=0)


def infer(windows_path: str, models_dir: str, threshold: float) -> pd.DataFrame:
    # 1. Load windowed data
    df = pd.read_parquet(windows_path)

    # 1a. Extract node_id, handling possible column names
    if 'node_id' in df.columns and not df['node_id'].isna().all():
        meta = df[['window_start', 'node_id']].copy()
    elif 'node' in df.columns and not df['node'].isna().all():
        meta = df[['window_start', 'node']].copy().rename(columns={'node': 'node_id'})
    else:
        meta = df[['window_start']].copy()
        meta['node_id'] = None

    # 1b. Drop metadata columns for feature matrix
    X = df.drop(columns=['window_start', 'node_id', 'node', 'y_cls', 'y_reg', 'delta'], errors='ignore')

    # 2. Load trained pipelines
    models_dir = Path(models_dir)
    cls_pipe = joblib.load(models_dir / 'cls.joblib')
    reg_pipe = joblib.load(models_dir / 'reg.joblib')

    # 3. Determine expected feature names from the imputer step
    imputer = cls_pipe.named_steps.get('impute')
    if imputer and hasattr(imputer, 'feature_names_in_'):
        expected_features = list(imputer.feature_names_in_)
    else:
        clf = cls_pipe.named_steps.get('clf')
        expected_features = list(getattr(clf, 'feature_names_in_', X.columns))

    # 4. Align feature DataFrame to expected columns
    X_aligned = align_features(X, expected_features)

    # 5. Perform predictions
    p_fail = cls_pipe.predict_proba(X_aligned)[:, 1]
    eta_min = reg_pipe.predict(X_aligned)

    # 6. Assemble results
    out = meta.copy()
    out['p_fail'] = p_fail
    out['eta_min'] = eta_min
    out['alert'] = (out['p_fail'] >= threshold).astype(int)

    # 7. Drop any rows where node_id is null or missing
    out = out[out['node_id'].notna()]

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows',   required=True, help='Parquet file of windowed data')
    parser.add_argument('--models',    required=True, help='Directory containing cls.joblib & reg.joblib')
    parser.add_argument('--threshold', type=float, default=0.5, help='Alert probability threshold')
    args = parser.parse_args()

    preds = infer(args.windows, args.models, args.threshold)
    preds.to_csv('predictions.csv', index=False)
    print('First rows of predictions.csv:')
    print(preds.head())
