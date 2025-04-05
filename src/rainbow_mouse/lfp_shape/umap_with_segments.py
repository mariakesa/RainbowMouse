import numpy as np
import pandas as pd
import umap
import plotly.express as px
import matplotlib.pyplot as plt
import io
import base64

# Load data
X = np.load('lfp_waveforms.npy')         # shape: [n_trials, features]
y = np.load('lfp_labels.npy') - 1        # shape: [n_trials]

# Reshape if needed
n_trials, total_len = X.shape
n_channels = 1  # <-- Adjust if your data isn't flattened
n_timepoints = total_len // n_channels
X_reshaped = X.reshape(n_trials, n_timepoints, n_channels).squeeze(-1)

# Convert waveforms to base64-encoded images
def waveform_to_base64(waveform):
    fig, ax = plt.subplots(figsize=(2, 0.5))
    ax.plot(waveform, color='black', linewidth=0.5)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}">'

hover_imgs = [waveform_to_base64(w) for w in X_reshaped]

# UMAP
reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, n_components=3, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X, y)

# DataFrame
df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2", "UMAP3"])
df["label"] = y
df["waveform_html"] = hover_imgs

# Now: use `custom_data` and `hovertemplate` explicitly
fig = px.scatter_3d(
    df,
    x="UMAP1", y="UMAP2", z="UMAP3",
    color=df["label"].astype(str),
)

# Inject waveform images into hover using `customdata`
fig.update_traces(
    marker=dict(size=4),
    customdata=np.stack([df["waveform_html"]], axis=-1),
    hovertemplate="%{customdata[0]}<extra></extra>",
)

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    title="Supervised 3D UMAP of LFP Waveforms with Hover Plots"
)

fig.show()
