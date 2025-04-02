import numpy as np
import torch


def sample_task_batch(lfp, frames, vit_embeddings, batch_size=128, device="cpu"):
    """
    Sample a random batch of (channel, time) LFP targets and corresponding ViT embeddings.

    Parameters:
    - lfp: np.array of shape [n_vit_frames=95, n_channels]
    - frames: np.array of shape [n_channels] -> values in 1..95
    - vit_embeddings: np.array of shape [95, 48]
    - batch_size: int
    - device: str (e.g., 'cuda' or 'cpu')

    Returns:
    - x_vit: Tensor of shape [B, 48]
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
    frame_ids = frames[sampled_channels] - 1
    vit_vecs = vit_embeddings[frame_ids]  # [B, 48]

    # Get LFP targets for the sampled (time, channel) pairs
    targets = lfp[sampled_channels, sampled_times]  # [B]

    return (
        torch.tensor(vit_vecs, dtype=torch.float32).to(device),
        torch.tensor(sampled_channels, dtype=torch.long).to(device),
        torch.tensor(targets, dtype=torch.float32).to(device)
    )
