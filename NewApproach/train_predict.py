# train_predict_cv.py
"""
Train and evaluate classification/regression with 5-fold cross-validation on windowed features.
Usage:
    python train_predict_cv.py --input data/bgl_windows_v1.parquet \
                                --models models/ --plots reports/
Outputs:
    - Fold-wise and average CV metrics for classification and regression
    - Final models (trained on full data) saved to models/
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
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor


def main(inp, models_dir, plots_dir):
    # Load data
    df = pd.read_parquet(inp)
    NONFEAT = {'window_start','node_id','y_cls','y_reg','delta'}
    X = df.drop(columns=NONFEAT)
    y_cls = df['y_cls'].values
    y_reg = df['y_reg'].values

    # Prepare CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cls_metrics = []
    reg_metrics = []

    # CV loop
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_cls), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]
        y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]

        # Classifier pipeline
        clf_pipe = Pipeline([
            ('imputer', SimpleImputer(fill_value=0)),
            ('smote',    SMOTE()),
            ('scale',    StandardScaler(with_mean=False)),
            ('clf',      XGBClassifier(n_jobs=-1, random_state=42))
        ])
        clf_pipe.fit(X_train, y_train_cls)
        # Predict
        y_pred_cls = clf_pipe.predict(X_test)
        # Metrics
        prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
        rec  = recall_score(y_test_cls, y_pred_cls, zero_division=0)
        f1   = f1_score(y_test_cls, y_pred_cls, zero_division=0)
        cls_metrics.append((prec, rec, f1))
        print(f"Fold {fold} classification -- precision: {prec:.3f}, recall: {rec:.3f}, f1: {f1:.3f}")

        # Regressor pipeline
        reg_pipe = Pipeline([
            ('imputer', SimpleImputer(fill_value=0)),
            ('reg',      XGBRegressor(n_jobs=-1, random_state=42))
        ])
        reg_pipe.fit(X_train, y_train_reg)
        y_pred_reg = reg_pipe.predict(X_test)
        # Metrics
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        r2  = r2_score(y_test_reg, y_pred_reg)
        reg_metrics.append((mae, r2))
        print(f"Fold {fold} regression -- MAE: {mae:.3f}, R2: {r2:.3f}\n")

    # Aggregate CV results
    cls_arr = np.array(cls_metrics)
    reg_arr = np.array(reg_metrics)
    print("=== Classification CV Averages ===")
    print(f"Precision: {cls_arr[:,0].mean():.3f} ± {cls_arr[:,0].std():.3f}")
    print(f"Recall:    {cls_arr[:,1].mean():.3f} ± {cls_arr[:,1].std():.3f}")
    print(f"F1-score:  {cls_arr[:,2].mean():.3f} ± {cls_arr[:,2].std():.3f}\n")
    print("=== Regression CV Averages ===")
    print(f"MAE: {reg_arr[:,0].mean():.3f} ± {reg_arr[:,0].std():.3f}")
    print(f"R2:  {reg_arr[:,1].mean():.3f} ± {reg_arr[:,1].std():.3f}\n")

    # Retrain on full data and save final models
    final_clf = Pipeline([
        ('imputer', SimpleImputer(fill_value=0)),
        ('smote',    SMOTE()),
        ('scale',    StandardScaler(with_mean=False)),
        ('clf',      XGBClassifier(n_jobs=-1, random_state=42))
    ])
    final_clf.fit(X, y_cls)
    joblib.dump(final_clf, Path(models_dir)/'cls_final.joblib')

    final_reg = Pipeline([
        ('imputer', SimpleImputer(fill_value=0)),
        ('reg',      XGBRegressor(n_jobs=-1, random_state=42))
    ])
    final_reg.fit(X, y_reg)
    joblib.dump(final_reg, Path(models_dir)/'reg_final.joblib')
    print("Saved final models as cls_final.joblib and reg_final.joblib")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--plots',  required=True)
    args = parser.parse_args()
    main(args.input, args.models, args.plots)
