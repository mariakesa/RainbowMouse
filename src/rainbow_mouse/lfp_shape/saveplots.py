# save_waveform_images.py
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
from tqdm import tqdm

# Load data
X = np.load('lfp_waveforms.npy')[:1000,:]  # shape: [n_trials, features]

# Create output dir
os.makedirs("waveform_imgs_base64", exist_ok=True)

# Convert waveform to base64 PNG string
def waveform_to_base64(waveform, out_path):
    fig, ax = plt.subplots(figsize=(2, 0.5))
    ax.plot(waveform, color='black', linewidth=0.5)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    with open(out_path, "w") as f:
        f.write(img_base64)

# Save all waveforms
for i, waveform in tqdm(enumerate(X), total=len(X)):
    out_file = f"waveform_imgs_base64/img_{i}.txt"
    if not os.path.exists(out_file):  # skip if already exists
        waveform_to_base64(waveform, out_file)

print("Done saving base64 waveform images.")
