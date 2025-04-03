import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# === Load test data ===
cache_path = os.environ.get("RAINBOW_MOUSE_CACHE")
lfp = np.load(f"{cache_path}/lfp_multi.npy")         # [95, 118*50]
frames = np.load(f"{cache_path}/frames_multi.npy") - 1   # [5900], now 0–117
vit_embeddings = np.load(f"{cache_path}/vit_embeddings.npy")  # [118, 192]

# Split train/test
lfp_train = lfp[:, :118*30]
lfp_test = lfp[:, 118*30:]
frames_train = frames[:118*30]
frames_test = frames[118*30:]

# Z-score using train stats
lfp_mean = np.mean(lfp_train, axis=1, keepdims=True)
lfp_std = np.std(lfp_train, axis=1, keepdims=True)

lfp_train = (lfp_train - lfp_mean) / lfp_std
lfp_test = (lfp_test - lfp_mean) / lfp_std

# === Load trained attention model ===
from rainbow_mouse.models.attention_multiprobe import CausalLFPTransformerMulti  # adjust path as needed

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CausalLFPTransformerMulti().to(device)
model.load_state_dict(torch.load(f"{cache_path}/lfp_multi.pth"))
model.eval()

# === Define parameters ===
window_size = model.window_size
half_window = window_size // 2
n_channels, n_trials = lfp_test.shape

# === Generate predictions ===
lfp_preds = np.zeros((n_channels, n_trials - window_size + 1))  # output is smaller due to window

print("Generating predictions for test set...")

for channel in tqdm(range(n_channels)):
    for t in range(half_window, n_trials - half_window):
        frame_window = frames_test[t - half_window : t + half_window + 1]  # [window_size]
        vit_window = vit_embeddings[frame_window]                          # [window_size, 192]
        vit_window_tensor = torch.tensor(vit_window, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, 192]
        channel_tensor = torch.tensor([channel], dtype=torch.long).to(device)

        with torch.no_grad():
            pred = model(vit_window_tensor, channel_tensor).cpu().item()

        lfp_preds[channel, t - half_window] = pred  # adjust index to match reduced output size

# === Plotting ===
def plot_lfp_heatmap(data, title):
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(label='Z-scored LFP')
    plt.title(title)
    plt.xlabel("Trial Index")
    plt.ylabel("Channel Index")
    plt.tight_layout()
    plt.show()
    
# Clip to same size for visual match
aligned_lfp_test = lfp_test[:, half_window : -half_window]

print("Mean of ground truth LFP (after z-scoring):", aligned_lfp_test.mean(axis=1))
plot_lfp_heatmap(aligned_lfp_test[:, :100], "Ground Truth LFPs (Test Set)")
plot_lfp_heatmap(lfp_preds[:, :100], "Predicted LFPs (Test Set)")

# === Compute variance explained per channel ===
r2_scores = []
for c in range(n_channels):
    y_true = aligned_lfp_test[c]
    y_pred = lfp_preds[c]
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2_scores.append(r2)

r2_scores = np.array(r2_scores)

# === Plot histogram of R² across channels ===
plt.figure(figsize=(8, 4))
plt.hist(r2_scores, bins=20, color='steelblue', edgecolor='black')
plt.axvline(np.mean(r2_scores), color='red', linestyle='--', label=f"Mean R² = {np.mean(r2_scores):.3f}")
plt.title("Distribution of Variance Explained (R²) per Channel")
plt.xlabel("R² Score")
plt.ylabel("Number of Channels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

