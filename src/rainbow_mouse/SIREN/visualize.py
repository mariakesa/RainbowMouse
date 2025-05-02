import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from rainbow_mouse.SIREN.lib import SIRENEncoder
import torch

def visualize_embedding_3d(data, encoder_path, input_dim, hidden_dim, output_dim,
                           num_layers=3, omega_0=30, title="3D Embedding of LFP"):
    encoder = SIRENEncoder(input_dim, hidden_dim, output_dim, num_layers, omega_0)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(torch.tensor(data, dtype=torch.float32))

    # If output_dim > 3, use PCA to reduce
    if output_dim > 3:
        print(f"Reducing {output_dim}D embedding to 3D using PCA")
        embeddings = PCA(n_components=3).fit_transform(embeddings.numpy())
    else:
        embeddings = embeddings.numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], color='mediumslateblue')
    ax.set_title(title)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    plt.tight_layout()
    plt.show()
