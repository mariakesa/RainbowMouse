# save_waveform_plots.py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Load your LFP waveform data
X = np.load('lfp_waveforms.npy')         # shape: [n_trials, features]
y = np.load('lfp_labels.npy') - 1        # optional: class labels

# Set up output directory
out_dir = "waveform_images"
os.makedirs(out_dir, exist_ok=True)

# Save each waveform as a PNG
for i, waveform in tqdm(enumerate(X), total=len(X)):
    label = y[i]
    out_path = os.path.join(out_dir, f"lfp_{i:05d}_label{label}.png")
    
    if not os.path.exists(out_path):  # skip if already saved
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.plot(waveform, color='black', linewidth=0.5)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

print(f"Saved {len(X)} waveform plots to: {out_dir}")
