import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# --- Define the Encoder Model ---
class LFPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, num_classes=118):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z  # return both for later use
# --- Load Data ---
X = torch.tensor(np.load('lfp_waveforms.npy'), dtype=torch.float32)
y = torch.tensor(np.load('lfp_labels.npy'), dtype=torch.long)
print("Labels:", y.min().item(), y.max().item())
# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# --- Model, Optimizer, Loss ---
model = LFPEncoder(input_dim=X.shape[1], embedding_dim=32, num_classes=len(y.unique()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
epochs = 20
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

# --- Evaluate & Extract Embeddings ---
model.eval()
all_embeddings, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        _, z = model(xb)
        all_embeddings.append(z.cpu())
        all_labels.append(yb)

all_embeddings = torch.cat(all_embeddings).numpy()
all_labels = torch.cat(all_labels).numpy()

# Save for UMAP
np.save("lfp_nn_embeddings.npy", all_embeddings)
np.save("lfp_nn_labels.npy", all_labels)
