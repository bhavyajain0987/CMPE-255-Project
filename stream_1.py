import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
from drain3 import TemplateMiner
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import umap

logging.getLogger("drain3").setLevel(logging.CRITICAL)

# Load & clean raw lines
LOG_FILE = "raw_data/BGL_2k.log"

lines = []
with open(LOG_FILE) as f:
    for line in f:
        raw = line.strip()
        # strip leading hyphens/spaces
        raw = re.sub(r'^[\-\s]+', '', raw)
        if raw:
            lines.append(raw)
print(f"Read {len(lines)} non-empty lines from {LOG_FILE}")

# Parse each line
pattern = re.compile(
    r'^(?P<epoch>\d+)\s+'              
    r'(?P<date>\S+)\s+'               
    r'(?P<node>\S+)\s+'               
    r'(?P<ts2>\S+)\s+'                
    r'(?P=node)\s+'                   
    r'RAS\s+(?P<component>\w+)\s+'    
    r'(?P<severity>\w+)\s+'           
    r'(?P<message>.+)$'               
)

records = []
for L in lines:
    m = pattern.match(L)
    if not m:
        continue
    d = m.groupdict()
    d['dt'] = pd.to_datetime(d['ts2'], format='%Y-%m-%d-%H.%M.%S.%f')
    records.append(d)

df = pd.DataFrame(records)
print(f"Parsed {len(df)} log entries.")

# Template extraction via Drain3
miner = TemplateMiner()
template_ids = []
for msg in df['message']:
    info = miner.add_log_message(msg)
    # accommodate either key name
    tid  = info.get("cluster_id", info.get("clusterId"))
    template_ids.append(tid)

df['template_id'] = template_ids

# Feature Engineering
# Text: TF-IDF on the raw message
tfidf = TfidfVectorizer(max_features=500)
X_text = tfidf.fit_transform(df['message'])

# Categorical: node, component, severity, template_id -> one-hot
X_cat = pd.get_dummies(
    df[['node','component','severity','template_id']].astype(str),
    drop_first=True
)
X_cat_sparse = sp.csr_matrix(X_cat.values)

# Temporal: inter-arrival (seconds) + hour-of-day
df['delta_s'] = df['dt'].diff().dt.total_seconds().fillna(0)
df['hour']    = df['dt'].dt.hour
X_time = sp.csr_matrix(df[['delta_s','hour']].values)

# Assemble & save feature matrix
X = sp.hstack([X_text, X_cat_sparse, X_time], format='csr')
print("Feature matrix shape:", X.shape)

# Records CSV (no message text)
df.drop(columns=['message']).to_csv("outputs/stream1_records.csv", index=False)

# Raw feature matrix (sparse .npz) + TFIDF meta
sp.save_npz("outputs/stream1_features.npz", X)
pd.DataFrame({'feature': tfidf.get_feature_names_out()}) \
  .to_csv("outputs/tfidf_features.csv", index=False)

# Dimensionality Reduction
pca = PCA(n_components=50)
Z50 = pca.fit_transform(X.toarray())

umapper = umap.UMAP(n_components=2, random_state=42)
Z2   = umapper.fit_transform(Z50)

# PCA‐50 CSV
pd.DataFrame(Z50, columns=[f"PC{i+1}" for i in range(50)]) \
  .to_csv("outputs/pca50.csv", index=False)

# UMAP‐2 CSV
pd.DataFrame(Z2, columns=["UMAP1","UMAP2"]) \
  .to_csv("outputs/umap2d.csv", index=False)

# Plot and save
plt.figure(figsize=(6,5))
plt.scatter(Z2[:,0], Z2[:,1], s=3, alpha=0.6)
plt.title("UMAP(PCA(X)) on BGL logs")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout()
plt.savefig("outputs/stream1_umap.png", dpi=150)
plt.show()

print("Stream 1 pipeline complete. Outputs in /outputs:")
print("  - stream1_records.csv")
print("  - stream1_features.npz + tfidf_features.csv")
print("  - pca50.csv, umap2d.csv")
print("  - stream1_umap.png")
