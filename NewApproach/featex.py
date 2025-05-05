"""
featex.py

Feature extraction and dimensionality reduction on windowed data,
with dynamic component selection for TruncatedSVD to avoid n_components errors.

Usage:
    python featex.py --input data/bgl_windows_v1.parquet
"""
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import umap
import matplotlib.pyplot as plt
import pathlib


def main(inp_path: str):
    # Load windowed data
    df = pd.read_parquet(inp_path)
    df.to_csv('outputs/stream1_records.csv', index=False)

    # Drop non-feature columns
    NONFEAT = {'window_start','node_id','y_cls','y_reg','delta'}
    X = df.drop(columns=NONFEAT, errors='ignore').values

    # Impute any missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imp = imputer.fit_transform(X)

    # Determine number of components for SVD
    n_features = X_imp.shape[1]
    n_comp = min(50, max(1, n_features - 1))
    print(f"Reducing to {n_comp} components (features: {n_features})")

    # 1) TruncatedSVD (handles sparse/large matrices)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z50 = svd.fit_transform(X_imp)
    pd.DataFrame(Z50, columns=[f'PC{i+1}' for i in range(n_comp)]) \
      .to_csv('outputs/pca50.csv', index=False)

    # 2) UMAP → 2D embedding
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z2 = reducer.fit_transform(Z50)
    pd.DataFrame(Z2, columns=['UMAP1','UMAP2']) \
      .to_csv('outputs/umap2d.csv', index=False)

    # 3) Plot UMAP
    plt.figure(figsize=(6,5))
    plt.scatter(Z2[:,0], Z2[:,1], c=df['y_cls'], cmap='coolwarm', s=3)
    plt.title('UMAP of Windowed Features (colored by y_cls)')
    plt.xlabel('UMAP1'); plt.ylabel('UMAP2')
    plt.tight_layout()
    pathlib.Path('outputs').mkdir(exist_ok=True)
    plt.savefig('outputs/stream1_umap.png', dpi=150)
    plt.close()

    print("Stream 1 features & embeddings saved to outputs/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Windowed Parquet file')
    args = parser.parse_args()
    main(args.input)
