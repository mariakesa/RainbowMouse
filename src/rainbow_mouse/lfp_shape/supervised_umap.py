import numpy as np
import pandas as pd
import umap
import plotly.express as px

# Load data
X = np.load('lfp_waveforms.npy')  # shape: [n_trials, features]
y = np.load('lfp_labels.npy') - 1  # assuming labels are 1-indexed and you want 0-indexed

# Supervised UMAP
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42
)
embedding = reducer.fit_transform(X, y)  # supervision added here

# Make DataFrame for Plotly
df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2", "UMAP3"])
df["label"] = y

# Plotly 3D scatter with label coloring
fig = px.scatter_3d(
    df, x="UMAP1", y="UMAP2", z="UMAP3",
    color=df["label"].astype(str),
    opacity=0.75,
    title="Supervised 3D UMAP of LFP Waveforms by Stimulus Label",
    labels={"color": "Label"}
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.show()
