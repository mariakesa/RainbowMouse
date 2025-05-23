import torch
import torch.nn as nn
import numpy as np

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first)

    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class SIRENEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, omega_0=30):
        super().__init__()
        layers = [SIRENLayer(input_dim, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TimeContrastiveSampler:
    def __init__(self, data, window_size=5, batch_size=128):
        self.data = data  # Shape: [T, D]
        self.window_size = window_size
        self.batch_size = batch_size
        self.T = data.shape[0]

    def sample(self):
        indices = np.random.randint(self.window_size, self.T - self.window_size, size=self.batch_size)
        ref = self.data[indices]
        pos = self.data[indices + np.random.randint(1, self.window_size + 1)]
        neg_indices = np.random.randint(0, self.T, size=self.batch_size)
        neg = self.data[neg_indices]
        return torch.tensor(ref, dtype=torch.float32), torch.tensor(pos, dtype=torch.float32), torch.tensor(neg, dtype=torch.float32)

import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_ref, z_pos, z_neg):
        z_ref = F.normalize(z_ref, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=1)

        pos_sim = torch.sum(z_ref * z_pos, dim=1) / self.temperature
        neg_sim = torch.sum(z_ref * z_neg, dim=1) / self.temperature

        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(z_ref.size(0), dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        return loss

import os
import torch

def train_cebra_time(data, input_dim, hidden_dim, output_dim, num_layers=3, omega_0=30,
                     epochs=1000, batch_size=128, learning_rate=1e-3, model_path="cebra_siren_encoder.pt"):
    encoder = SIRENEncoder(input_dim, hidden_dim, output_dim, num_layers, omega_0)
    sampler = TimeContrastiveSampler(data, window_size=5, batch_size=batch_size)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        ref, pos, neg = sampler.sample()
        z_ref = encoder(ref)
        z_pos = encoder(pos)
        z_neg = encoder(neg)

        loss = criterion(z_ref, z_pos, z_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(encoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return encoder

import torch.nn as nn

class ReLUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_cebra_time_relu(data, input_dim, hidden_dim, output_dim, num_layers=3,
                          epochs=1000, batch_size=128, learning_rate=1e-3,
                          model_path="cebra_relu_encoder.pt"):
    encoder = ReLUEncoder(input_dim, hidden_dim, output_dim, num_layers)
    sampler = TimeContrastiveSampler(data, window_size=5, batch_size=batch_size)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        ref, pos, neg = sampler.sample()
        z_ref = encoder(ref)
        z_pos = encoder(pos)
        z_neg = encoder(neg)

        loss = criterion(z_ref, z_pos, z_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[ReLU] Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(encoder.state_dict(), model_path)
    print(f"ReLU model saved to {model_path}")

    return encoder

import torch
import torch.nn as nn

class MultiFreqSIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, frequencies, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.frequencies = frequencies
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = 6 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        base = self.linear(x)
        components = [torch.sin(ω * base) for ω in self.frequencies]
        return torch.cat(components, dim=-1)

class MultiFreqSIRENEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 frequencies=[5.0, 10.0, 30.0, 60.0], num_layers=3):
        super().__init__()
        self.frequencies = frequencies
        freq_count = len(frequencies)

        layers = [
            MultiFreqSIRENLayer(input_dim, hidden_dim, frequencies, is_first=True),
        ]
        for _ in range(num_layers - 2):
            layers.append(MultiFreqSIRENLayer(hidden_dim * freq_count, hidden_dim, frequencies))

        layers.append(nn.Linear(hidden_dim * freq_count, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_cebra_time_multifreq_siren(data, input_dim, hidden_dim, output_dim,
                                     frequencies=[5, 10, 30, 60], num_layers=3,
                                     epochs=1000, batch_size=128, learning_rate=1e-3,
                                     model_path="cebra_multifreq_siren.pt"):
    encoder = MultiFreqSIRENEncoder(input_dim, hidden_dim, output_dim,
                                    frequencies=frequencies, num_layers=num_layers)
    sampler = TimeContrastiveSampler(data, window_size=5, batch_size=batch_size)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        ref, pos, neg = sampler.sample()
        z_ref = encoder(ref)
        z_pos = encoder(pos)
        z_neg = encoder(neg)

        loss = criterion(z_ref, z_pos, z_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[MultiFreq SIREN] Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(encoder.state_dict(), model_path)
    print(f"Multi-frequency SIREN model saved to {model_path}")

    return encoder

def visualize_embedding_3d_multifreq_siren(data, encoder_path, input_dim, hidden_dim, output_dim,
                                           frequencies=[5, 10, 30, 60], num_layers=3,
                                           title="3D Multi-Frequency SIREN Embedding of LFP"):
    encoder = MultiFreqSIRENEncoder(input_dim, hidden_dim, output_dim,
                                    frequencies=frequencies, num_layers=num_layers)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(torch.tensor(data, dtype=torch.float32))

    # PCA if needed
    if output_dim > 3:
        from sklearn.decomposition import PCA
        print(f"Reducing {output_dim}D embedding to 3D using PCA")
        embeddings = PCA(n_components=3).fit_transform(embeddings.numpy())
    else:
        embeddings = embeddings.numpy()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], color='seagreen')
    ax.set_title(title)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    plt.tight_layout()
    plt.show()


import matplotlib
matplotlib.use("TkAgg")  # Enables interactive GUI windows

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from sklearn.decomposition import PCA
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving animations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.decomposition import PCA
import torch
import numpy as np

def animate_embedding_3d_multifreq_siren(data, encoder_path, input_dim, hidden_dim, output_dim,
                                         frequencies=[5, 10, 30, 60], num_layers=3,
                                         title="3D Embedding Animation", save_path="embedding_rotation.mp4"):
    # Load trained model
    encoder = MultiFreqSIRENEncoder(input_dim, hidden_dim, output_dim,
                                    frequencies=frequencies, num_layers=num_layers)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(torch.tensor(data, dtype=torch.float32))

    # Reduce to 3D if needed
    if output_dim > 3:
        embeddings = PCA(n_components=3).fit_transform(embeddings.numpy())
    else:
        embeddings = embeddings.numpy()

    # Setup figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")

    # Plot static points
    scatter = ax.plot(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], color='seagreen')[0]

    # Rotation update function
    def update(angle):
        ax.view_init(elev=30, azim=angle)
        return fig,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)

    # Save as mp4 or gif
    if save_path.endswith(".gif"):
        ani.save(save_path, writer="pillow", fps=20)
    else:
        ani.save(save_path, writer="ffmpeg", fps=20)

    print(f"Saved rotating animation to {save_path}")


import torch
import torch.nn as nn

class LearnableFreqSIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=4, is_first=False, omega_init_range=(1.0, 60.0)):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first

        # Learnable frequencies: shape [num_frequencies]
        low, high = omega_init_range
        self.omega = nn.Parameter(
            torch.linspace(low, high, steps=num_frequencies).view(1, 1, -1)
        )  # shape [1, 1, K]

        self._init_weights()

        self.out_features = out_features
        self.num_frequencies = num_frequencies

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = 6 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        """
        x: [B, in_features]
        output: [B, out_features * num_frequencies]
        """
        z = self.linear(x)  # shape: [B, out_features]
        z = z.unsqueeze(-1)  # [B, out_features, 1]

        # Apply learned frequencies
        out = torch.sin(self.omega * z)  # [B, out_features, K]
        return out.view(x.shape[0], -1)  # flatten to [B, out_features * K]

class LearnableFreqSIRENEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=3, num_frequencies=4, omega_init_range=(1.0, 60.0)):
        super().__init__()

        layers = [
            LearnableFreqSIRENLayer(input_dim, hidden_dim, num_frequencies, is_first=True, omega_init_range=omega_init_range)
        ]

        for _ in range(num_layers - 2):
            layers.append(
                LearnableFreqSIRENLayer(hidden_dim * num_frequencies, hidden_dim,
                                        num_frequencies, is_first=False, omega_init_range=omega_init_range)
            )

        layers.append(nn.Linear(hidden_dim * num_frequencies, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_cebra_time_learnable_siren(
    data,
    input_dim,
    hidden_dim,
    output_dim,
    num_layers=3,
    num_frequencies=4,
    omega_init_range=(1.0, 60.0),
    epochs=1000,
    batch_size=128,
    learning_rate=1e-3,
    model_path="cebra_learnable_siren.pt"
):
    from rainbow_mouse.SIREN.lib import TimeContrastiveSampler, ContrastiveLoss  # or adjust import if local

    encoder = LearnableFreqSIRENEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        omega_init_range=omega_init_range
    )

    sampler = TimeContrastiveSampler(data, window_size=5, batch_size=batch_size)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        ref, pos, neg = sampler.sample()
        z_ref = encoder(ref)
        z_pos = encoder(pos)
        z_neg = encoder(neg)

        loss = criterion(z_ref, z_pos, z_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[LearnableFreqSIREN] Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(encoder.state_dict(), model_path)
    print(f"Learnable frequency SIREN model saved to {model_path}")

    return encoder

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def animate_embedding_3d_learnable_siren(data, encoder_path, input_dim, hidden_dim, output_dim,
                                         num_layers=3, num_frequencies=6,
                                         omega_init_range=(1.0, 60.0),
                                         title="3D Embedding Animation (Learnable ω)",
                                         save_path="embedding_rotation_learnable.mp4"):
    # Load trained LearnableFreqSIRENEncoder with correct init config
    encoder = LearnableFreqSIRENEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        omega_init_range=omega_init_range
    )
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    # Encode the data
    with torch.no_grad():
        embeddings = encoder(torch.tensor(data, dtype=torch.float32))

    # Reduce to 3D if needed
    if output_dim > 3:
        embeddings = PCA(n_components=3).fit_transform(embeddings.numpy())
    else:
        embeddings = embeddings.numpy()

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")

    # Plot initial points
    scatter = ax.plot(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], '.', color='indigo')[0]

    # Update function for rotation
    def update(angle):
        ax.view_init(elev=30, azim=angle)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)

    # Save the animation
    if save_path.endswith(".gif"):
        ani.save(save_path, writer="pillow", fps=20)
    else:
        ani.save(save_path, writer="ffmpeg", fps=20)

    print(f"Saved rotating embedding animation to {save_path}")