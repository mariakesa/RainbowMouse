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
