import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import os
from rainbow_mouse.learning.utils import sample_task_batch_context
from dotenv import load_dotenv
load_dotenv()

def train_model(model, lfp, frames, vit_embeddings, 
                device="cuda", batch_size=1024, 
                n_steps=100000, log_interval=10000, 
                lr=1e-3, cache_path=None):
    
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_log = []

    for step in tqdm(range(n_steps)):
        # Sample batch
        x_vit_window, channel_idx, y_target = sample_task_batch_context(
            lfp=lfp, frames=frames, vit_embeddings=vit_embeddings,
            batch_size=batch_size, device=device, window_size=model.window_size
        )
        y_pred = model(x_vit_window, channel_idx)
        # Forward

        # Loss
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_log.append(loss.item())

        if step % log_interval == 0 or step == n_steps - 1:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # Save model
    if cache_path:
        PATH = os.path.join(cache_path, "lfp_model_attention.pth")
        torch.save(model.state_dict(), PATH)
        print(f"Model saved to {PATH}")

    return loss_log

from rainbow_mouse.models.attention import CausalLFPTransformer  # or define in notebook

# Load and z-score data
cache_path = os.environ.get("RAINBOW_MOUSE_CACHE")
lfp = np.load(f"{cache_path}/lfp_X.npy")[:, :118*30]         # [95, 5900]
frames = np.load(f"{cache_path}/lfp_y.npy")[:118*30] - 1     # [5900]
vit_embeddings = np.load(f"{cache_path}/vit_embeddings.npy") # [95, 192]

# Normalize LFPs
lfp = (lfp - np.mean(lfp, axis=1, keepdims=True)) / np.std(lfp, axis=1, keepdims=True)


# Model
model = CausalLFPTransformer()

# Train
loss_log = train_model(model, lfp, frames, vit_embeddings, 
                       device="cuda", cache_path=cache_path)