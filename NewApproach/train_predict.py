# train_predict.py

"""
Train & evaluate classification/regression with 5-fold CV on PCA-reduced features.
Usage:
    python train_predict.py \
      --records outputs/stream1_records.csv \
      --features outputs/pca50.csv \
      --models models/ \
      --plots reports/

Inputs:
  --records   CSV of windows with y_cls & y_reg (from featex.py)
  --features  CSV of PCA components (from featex.py)
Outputs:
  - Fold-wise confusion matrices, precision, recall, F1, MAE, R² printed
  - Classification CV averages (precision, recall, F1)
  - Regression CV averages (MAE, R²)
  - Final models saved as cls_final.joblib and reg_final.joblib in <models>/
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

def main(records_csv, features_csv, models_dir, plots_dir):
    # Load inputs
    df_win = pd.read_csv(records_csv, parse_dates=['window_start'])
    df_feat = pd.read_csv(features_csv)

    if len(df_win) != len(df_feat):
        raise ValueError("records and features row counts differ")

    X = df_feat.values
    y_cls = df_win['y_cls'].values
    y_reg = df_win['y_reg'].values

    # Prepare CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cls_metrics = []
    reg_metrics = []

    # 5-fold CV
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_cls), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]
        y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]

        # Classification pipeline
        clf_pipe = Pipeline([
            ('impute', SimpleImputer(fill_value=0)),
            ('scale',  StandardScaler()),
            ('smote',  SMOTE(random_state=42)),
            ('clf',    XGBClassifier(n_jobs=-1, random_state=42))
        ])
        clf_pipe.fit(X_train, y_train_cls)
        y_pred_cls = clf_pipe.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test_cls, y_pred_cls)
        print(f"Fold {fold} Confusion Matrix:\n{cm}")

        # Metrics
        prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
        rec  = recall_score(y_test_cls, y_pred_cls, zero_division=0)
        f1   = f1_score(y_test_cls, y_pred_cls, zero_division=0)
        cls_metrics.append((prec, rec, f1))
        print(f"Fold {fold} Classification — precision: {prec:.3f}, recall: {rec:.3f}, f1: {f1:.3f}\n")

        # Regression pipeline
        reg_pipe = Pipeline([
            ('impute', SimpleImputer(fill_value=0)),
            ('reg',    XGBRegressor(n_jobs=-1, random_state=42))
        ])
        reg_pipe.fit(X_train, y_train_reg)
        y_pred_reg = reg_pipe.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        r2  = r2_score(y_test_reg, y_pred_reg)
        reg_metrics.append((mae, r2))
        print(f"Fold {fold} Regression — MAE: {mae:.3f}, R²: {r2:.3f}\n")

    # Aggregate CV results
    cls_arr = np.array(cls_metrics)
    reg_arr = np.array(reg_metrics)

    print("=== Classification CV Averages ===")
    print(f"Precision: {cls_arr[:,0].mean():.3f} ± {cls_arr[:,0].std():.3f}")
    print(f"Recall:    {cls_arr[:,1].mean():.3f} ± {cls_arr[:,1].std():.3f}")
    print(f"F1-score:  {cls_arr[:,2].mean():.3f} ± {cls_arr[:,2].std():.3f}\n")

    print("=== Regression CV Averages ===")
    print(f"MAE: {reg_arr[:,0].mean():.3f} ± {reg_arr[:,0].std():.3f}")
    print(f"R²:  {reg_arr[:,1].mean():.3f} ± {reg_arr[:,1].std():.3f}\n")

    # Retrain on full data & save final models
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    final_clf = Pipeline([
        ('impute', SimpleImputer(fill_value=0)),
        ('scale',  StandardScaler()),
        ('smote',  SMOTE(random_state=42)),
        ('clf',    XGBClassifier(n_jobs=-1, random_state=42))
    ])
    final_clf.fit(X, y_cls)
    joblib.dump(final_clf, Path(models_dir) / 'cls_final.joblib')

    final_reg = Pipeline([
        ('impute', SimpleImputer(fill_value=0)),
        ('reg',    XGBRegressor(n_jobs=-1, random_state=42))
    ])
    final_reg.fit(X, y_reg)
    joblib.dump(final_reg, Path(models_dir) / 'reg_final.joblib')

    print("✅ Final models saved: cls_final.joblib, reg_final.joblib")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records',  required=True,
                        help='stream1_records.csv from featex.py')
    parser.add_argument('--features', required=True,
                        help='pca50.csv from featex.py')
    parser.add_argument('--models',   required=True,
                        help='Directory to save final models')
    parser.add_argument('--plots',    required=True,
                        help='(unused, but kept for consistency)')
    args = parser.parse_args()
    main(args.records, args.features, args.models, args.plots)