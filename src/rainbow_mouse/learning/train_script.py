import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
from utils import sample_task_batch
from rainbow_mouse.models.first_model import LFPChannelEmbeddingModel
import torch.optim as optim
from tqdm import tqdm
# === TRAINING SCRIPT ===

def train_model(
    lfp, frames, vit_embeddings,
    n_channels=95,
    batch_size=128,
    device="cuda",
    epochs=1000,
    print_every=10
):
    model = LFPChannelEmbeddingModel(n_channels=n_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        x_vit, channel_idx, y = sample_task_batch(lfp, frames, vit_embeddings, batch_size, device)
        y_pred = model(x_vit, channel_idx)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    return model

# === MAIN ===
if __name__ == "__main__":
    cache_path = os.environ.get("RAINBOW_MOUSE_CACHE")
    lfp = np.load(f"{cache_path}/lfp_X.npy")[:,:118*30]         # [95, 5900]
    #print('badaboom', lfp.shape)
    frames = np.load(f"{cache_path}/lfp_y.npy")[:118*30]     # [5900]
    print(lfp.shape, frames.shape)
    vit_embeddings = np.load(f"{cache_path}/vit_embeddings.npy")  # [95, 48]

    print('Zscore mean', np.mean(lfp, axis=1), 'std', np.std(lfp, axis=0))

    # Normalize LFP data
    lfp = (lfp - np.mean(lfp, axis=1, keepdims=True)) / np.std(lfp, axis=1, keepdims=True)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(lfp, frames, vit_embeddings, device=device)
    PATH= f"{cache_path}/lfp_model.pth"
    torch.save(model.state_dict(), PATH)
