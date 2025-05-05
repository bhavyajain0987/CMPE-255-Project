# featex.py

"""
Feature extraction, standardization, and dimensionality reduction on windowed data.
Usage:
    python featex.py --input data/bgl_windows_v1.parquet
Outputs in outputs/:
  - stream1_records.csv
  - feature_stats.csv   (means rounded to 8 decimals, stds to 8)
  - pca50.csv
  - umap2d.csv
  - stream1_umap.png
"""
 
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import umap
import matplotlib.pyplot as plt
import pathlib

def main(inp_path: str):
    # 1) Load & save raw windows
    df = pd.read_parquet(inp_path)
    pathlib.Path('outputs').mkdir(exist_ok=True)
    df.to_csv('outputs/stream1_records.csv', index=False)

    # 2) Drop non-feature columns
    NONFEAT = {'window_start','node_id','y_cls','y_reg','delta'}
    X_df = df.drop(columns=NONFEAT, errors='ignore')
    feat_names = X_df.columns.tolist()

    # 3) Impute missing → zeros
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imp = imputer.fit_transform(X_df)

    # 4) Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # 5) Compute & save feature stats (rounded for readability)
    means = X_scaled.mean(axis=0)
    stds  = X_scaled.std(axis=0)
    stats = pd.DataFrame({
        'feature': feat_names,
        'mean':    means,
        'std':     stds
    })
    # Round to 8 decimal places for display
    stats[['mean','std']] = stats[['mean','std']].round(8)
    stats.to_csv('outputs/feature_stats.csv', index=False)
    print("Saved feature_stats.csv (means ~0, stds ~1, rounded)")

    # 6) TruncatedSVD → <=50 components
    n_features = X_scaled.shape[1]
    n_comp = min(50, max(1, n_features - 1))
    print(f"Reducing {n_features} features → {n_comp} SVD components")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z50 = svd.fit_transform(X_scaled)
    pd.DataFrame(Z50, columns=[f'PC{i+1}' for i in range(n_comp)]) \
      .to_csv('outputs/pca50.csv', index=False)

    # 7) UMAP → 2D embedding
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z2 = reducer.fit_transform(Z50)
    pd.DataFrame(Z2, columns=['UMAP1','UMAP2']) \
      .to_csv('outputs/umap2d.csv', index=False)

    # 8) Plot UMAP colored by true label
    plt.figure(figsize=(6,5))
    plt.scatter(Z2[:,0], Z2[:,1], c=df['y_cls'], cmap='coolwarm', s=3)
    plt.title('UMAP of Windowed Features (y_cls)')
    plt.xlabel('UMAP1'); plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig('outputs/stream1_umap.png', dpi=150)
    plt.close()

    print("featex.py complete. All outputs in outputs/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Windowed Parquet file')
    args = parser.parse_args()
    main(args.input)