import torch
import torch.nn as nn

class CausalLFPTransformer(nn.Module):
    def __init__(self, 
                 n_channels=95, 
                 image_embed_dim=192,   # 3 × 64D ViTs
                 channel_embed_dim=64, 
                 attn_heads=4,
                 hidden_dim=256,
                 window_size=5):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.image_embed_dim = image_embed_dim
        self.channel_embed_dim = channel_embed_dim
        
        # Channel embedding (query)
        self.channel_embed = nn.Embedding(n_channels, channel_embed_dim)

        # Project channel embedding to same dim as image embeddings
        self.query_proj = nn.Linear(channel_embed_dim, image_embed_dim)

        # Positional encoding: one learnable vector per time step
        self.positional_embed = nn.Parameter(torch.randn(window_size, image_embed_dim))

        # Multihead attention
        self.attn = nn.MultiheadAttention(embed_dim=image_embed_dim, 
                                          num_heads=attn_heads, 
                                          batch_first=True)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_window, channel_idx):
        """
        image_window: [B, T, 192] – past image embeddings
        channel_idx: [B] – int64 tensor with channel indices
        """
        B, T, D = image_window.shape
        assert T == self.window_size, f"Expected window size {self.window_size}, got {T}"

        # Add positional encoding
        pos = self.positional_embed.unsqueeze(0)          # [1, T, D]
        image_window = image_window + pos                 # [B, T, D]

        # Channel → query vector
        c_embed = self.channel_embed(channel_idx)         # [B, 64]
        q = self.query_proj(c_embed).unsqueeze(1)         # [B, 1, D]

        # Attention: query is channel embedding, attends to window of frames
        attn_out, _ = self.attn(q, image_window, image_window)  # [B, 1, D]

        # Predict LFP from attended context
        y_hat = self.mlp(attn_out.squeeze(1)).squeeze(-1)       # [B]

        return y_hat
