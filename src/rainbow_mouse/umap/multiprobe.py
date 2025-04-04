from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import umap
import plotly.express as px

# --- Load data ---
load_dotenv()
rainbow_cache = os.getenv("RAINBOW_MOUSE_CACHE")

lfp = np.load(rainbow_cache + "/lfp_multi.npy").T       # shape: [n_trials, n_channels]
labels = np.load(rainbow_cache + "/frames_multi.npy")  # shape: [n_trials]

# --- Replace NaNs with small noise ---
eta = 1e-6
nan_mask = np.isnan(lfp)
lfp[nan_mask] = np.random.normal(loc=0.0, scale=eta, size=nan_mask.sum())

# --- Run UMAP directly on LFPs ---
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=3, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(lfp)

# --- Create DataFrame for Plotly ---
df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2", "UMAP3"])
df["label"] = labels.astype(str)

# --- Plot ---
fig = px.scatter_3d(
    df, x="UMAP1", y="UMAP2", z="UMAP3",
    color="label",
    title="3D UMAP of Mean LFP per Trial (Unmodified)",
    opacity=0.8
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.show()
