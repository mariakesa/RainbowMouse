import numpy as np
import torch


def sample_task_batch(lfp, frames, vit_embeddings, batch_size=1000, device="cpu"):
    """
    Sample a random batch of (channel, time) LFP targets and corresponding ViT embeddings.

    Parameters:
    - lfp: np.array of shape [n_vit_frames=time_points, n_channels]
    - frames: np.array of shape [n_timepoints]
    - vit_embeddings: np.array of shape 
    - batch_size: int
    - device: str (e.g., 'cuda' or 'cpu')

    Returns:
    - x_vit: Tensor of shape = [B, n*3] (float)
    - channel_idx: Tensor of shape [B] (long)
    - y_target: Tensor of shape [B] (float)
    """
    #print('Boom ', lfp.shape)
    n_channels = lfp.shape[0]
    time_points = lfp.shape[1]

    # Sample random (channel, time) pairs
    sampled_channels = np.random.randint(0, n_channels, size=batch_size)
    sampled_times = np.random.randint(0, time_points, size=batch_size)

    # Get the frame ID for each channel (map from 1–95 → 0–94)
    frame_ids = frames[sampled_times] 
    vit_vecs = vit_embeddings[frame_ids]  # [B, 48]

    # Get LFP targets for the sampled (time, channel) pairs
    targets = lfp[sampled_channels, sampled_times]  # [B]
    #print(targets)

    return (
        torch.tensor(vit_vecs, dtype=torch.float32).to(device),
        torch.tensor(sampled_channels, dtype=torch.long).to(device),
        torch.tensor(targets, dtype=torch.float32).to(device)
    )

def sample_task_batch_context(lfp, frames, vit_embeddings, 
                              batch_size=1000, device="cpu", 
                              window_size=5):
    """
    Returns:
    - x_vit_window: Tensor of shape [B, window_size, 192]
    - channel_idx: Tensor of shape [B]
    - y_target: Tensor of shape [B]
    """
    n_channels, n_timepoints = lfp.shape
    half_w = window_size // 2

    # Keep only timepoints where full window fits
    valid_times = np.arange(half_w, n_timepoints - half_w)
    sampled_times = np.random.choice(valid_times, size=batch_size)
    sampled_channels = np.random.randint(0, n_channels, size=batch_size)

    x_vit_window = []
    y_target = []
    
    for t, ch in zip(sampled_times, sampled_channels):
        frame_window = frames[t - half_w : t + half_w + 1]  # shape [window_size]
        vit_window = vit_embeddings[frame_window]           # [window_size, 192]
        x_vit_window.append(vit_window)
        y_target.append(lfp[ch, t])                         # scalar target

    x_vit_window = torch.tensor(np.stack(x_vit_window), dtype=torch.float32).to(device)  # [B, T, 192]
    channel_idx = torch.tensor(sampled_channels, dtype=torch.long).to(device)            # [B]
    y_target = torch.tensor(y_target, dtype=torch.float32).to(device)                    # [B]

    return x_vit_window, channel_idx, y_target

