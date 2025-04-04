import umap
import numpy as np
import matplotlib.pyplot as plt

# Load data
X= np.load('lfp_waveforms.npy')  # Assuming the data is in a numpy array format

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)

# Plot
#plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c='steelblue', s=10, alpha=0.7)
plt.title("UMAP of LFP Waveforms per Trial")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.tight_layout()
plt.show()