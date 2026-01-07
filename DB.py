import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Clustering - DBSCAN",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Clustering Analysis (DBSCAN)")
st.caption("Loaded from trained DBSCAN artifacts (joblib)")

# --------------------------------------------------
# LOAD DATA & TRAINED FILES (NO CACHE ISSUES)
# --------------------------------------------------
def load_assets():
    scaler = joblib.load("wine_scaler.pkl")
    X_scaled = joblib.load("X_scaled_training.pkl")
    labels = joblib.load("dbscan_labels.pkl")
    data = pd.read_csv(r"C:\Users\NandhaKishore\Desktop\New Project\DATA_SETS\wine_clustering_data.csv")

    labels = np.array(labels).astype(int)
    X_scaled = np.array(X_scaled)

    return scaler, X_scaled, labels, data

scaler, X_scaled, labels, df = load_assets()

# --------------------------------------------------
# SAFETY CHECK
# --------------------------------------------------
if len(labels) != len(df):
    st.error("❌ ERROR: Cluster labels length does not match dataset")
    st.stop()

# --------------------------------------------------
# ADD CLUSTER COLUMN
# --------------------------------------------------
df["Cluster"] = labels

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# CLUSTER SUMMARY (YOUR EXACT REQUIREMENT)
# --------------------------------------------------
unique_clusters = sorted(set(labels))
clusters_only = [c for c in unique_clusters if c != -1]
noise_points = int(np.sum(labels == -1))

st.subheader("📌 DBSCAN Clustering Result")

st.success(f"Clusters found: {', '.join(map(str, clusters_only))}")
st.warning(f"Noise points: {noise_points}")

# --------------------------------------------------
# CLUSTER COUNTS
# --------------------------------------------------
st.subheader("📊 Points per Cluster")

cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
    .rename("Number of Samples")
)

st.dataframe(cluster_counts)

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------
st.subheader("📈 PCA Cluster Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

fig1, ax1 = plt.subplots(figsize=(8, 6))

for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    if cluster == -1:
        ax1.scatter(
            subset["PCA1"],
            subset["PCA2"],
            c="red",
            marker="x",
            label="Noise"
        )
    else:
        ax1.scatter(
            subset["PCA1"],
            subset["PCA2"],
            label=f"Cluster {cluster}"
        )

ax1.set_xlabel("PCA Component 1")
ax1.set_ylabel("PCA Component 2")
ax1.set_title("DBSCAN Clustering (PCA)")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# --------------------------------------------------
# ALCOHOL vs MALIC ACID PLOT (YOUR PLOT FIXED)
# --------------------------------------------------
st.subheader("🍷 Alcohol vs Malic Acid (DBSCAN Clusters)")

fig2, ax2 = plt.subplots(figsize=(10, 6))

scatter = ax2.scatter(
    df["alcohol"],
    df["malic_acid"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.7
)

ax2.set_title("DBSCAN: Alcohol vs Malic Acid")
ax2.set_xlabel("Alcohol")
ax2.set_ylabel("Malic Acid")
plt.colorbar(scatter, ax=ax2, label="Cluster ID (-1 = Noise)")
ax2.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig2)

# --------------------------------------------------
# DOWNLOAD CLUSTERED DATA
# --------------------------------------------------
st.subheader("⬇️ Download Clustered Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="wine_dbscan_clustered.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("DBSCAN clustering analysis | No retraining | Joblib-based")
