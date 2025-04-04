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

# Get number of channels and timepoints if you know it
n_trials, total_len = X.shape
n_channels = 1                            # <-- CHANGE THIS if needed
n_timepoints = total_len // n_channels
X_reshaped = X.reshape(n_trials, n_timepoints, n_channels).squeeze(-1)  # shape: [n_trials, n_timepoints]

# Create small inline plots for each waveform
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
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=3, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X, y)

# DataFrame
df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2", "UMAP3"])
df["label"] = y
df["waveform_html"] = hover_imgs

# Plotly
fig = px.scatter_3d(
    df,
    x="UMAP1", y="UMAP2", z="UMAP3",
    color=df["label"].astype(str),
    hover_data={"waveform_html": True, "label": True, "UMAP1": False, "UMAP2": False, "UMAP3": False},
    title="Supervised 3D UMAP of LFP Waveforms with Hover Plots"
)

# Allow raw HTML in hover tooltips
fig.update_traces(marker=dict(size=4), hovertemplate="<br>%{customdata[0]}<extra></extra>")
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

fig.show()
