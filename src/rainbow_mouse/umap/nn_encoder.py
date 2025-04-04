import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv

# --- Load data ---
load_dotenv()
rainbow_cache = os.getenv("RAINBOW_MOUSE_CACHE")

lfp = np.load(rainbow_cache + "/lfp_multi.npy").T        # shape: [n_trials, n_channels]
labels = np.load(rainbow_cache + "/frames_multi.npy")  # shape: [n_trials]

# --- Clean NaNs ---
eta = 1e-6
nan_mask = np.isnan(lfp)
lfp[nan_mask] = np.random.normal(loc=0.0, scale=eta, size=nan_mask.sum())

# --- Encode labels to contiguous 0-based indices ---
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(lfp, labels_encoded, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# --- Define model ---
class LFPDecoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, num_classes=118):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z

# --- Train model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LFPDecoder(input_dim=lfp.shape[1], embedding_dim=32, num_classes=len(label_encoder.classes_)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()

    train_acc = correct / len(train_ds)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f}")

# --- Extract embeddings from test set ---
model.eval()
all_embeddings = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        _, z = model(xb)
        all_embeddings.append(z.cpu())
        all_labels.append(yb)

all_embeddings = torch.cat(all_embeddings).numpy()
all_labels = torch.cat(all_labels).numpy()

# --- Save for UMAP ---
np.save("lfp_nn_flat_embeddings.npy", all_embeddings)
np.save("lfp_nn_flat_labels.npy", all_labels)
