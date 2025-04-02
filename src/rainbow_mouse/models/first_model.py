import torch
import torch.nn as nn
import torch.nn.functional as F

class LFPChannelEmbeddingModel(nn.Module):
    def __init__(self, n_channels=95, vit_dim=48, channel_dim=16, hidden_dim=64):
        super().__init__()
        self.channel_embed = nn.Embedding(n_channels, channel_dim)

        self.mlp = nn.Sequential(
            nn.Linear(vit_dim + channel_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output: scalar LFP
        )

    def forward(self, x, channel_idx):
        """
        x: Tensor of shape [B, 48] - ViT embeddings
        channel_idx: LongTensor of shape [B] - indices into channel embedding
        """
        e = self.channel_embed(channel_idx)  # shape: [B, 16]
        inp = torch.cat([x, e], dim=-1)      # shape: [B, 64]
        y_hat = self.mlp(inp).squeeze(-1)    # shape: [B]
        return y_hat
