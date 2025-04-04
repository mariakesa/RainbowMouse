import numpy as np
import pandas as pd
import umap
import plotly.express as px

# --- Load learned embeddings and labels ---
X = np.load('lfp_nn_embeddings.npy')  # shape: [n_samples, embedding_dim]
y = np.load('lfp_nn_labels.npy')      # shape: [n_samples]

print(X.shape)
# --- Run 3D UMAP ---
reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, n_components=3, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)  # shape: [n_samples, 3]

# --- Create DataFrame for Plotly ---
df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2', 'UMAP3'])
df['label'] = y.astype(str)  # use string labels for color

# --- Plot with Plotly ---
fig = px.scatter_3d(
    df, x='UMAP1', y='UMAP2', z='UMAP3',
    color='label',
    opacity=0.75,
    title='3D UMAP of Neural Network LFP Embeddings',
    labels={'label': 'Stimulus'}
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.show()
