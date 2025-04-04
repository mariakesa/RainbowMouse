import umap
import numpy as np
import plotly.express as px

# Load data
X = np.load('lfp_waveforms.npy')  # shape: [n_trials, features]

# Reduce to 3 dimensions
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean', n_components=3, random_state=42)
embedding = reducer.fit_transform(X)  # shape: [n_trials, 3]

# Convert to DataFrame for Plotly
import pandas as pd
df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2', 'UMAP3'])

# Plotly 3D scatter
fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3',
                    opacity=0.7,
                    color_discrete_sequence=['steelblue'],
                    title="3D UMAP of LFP Waveforms per Trial")

fig.update_traces(marker=dict(size=3))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.show()
