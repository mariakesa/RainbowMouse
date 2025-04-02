import pickle
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

data_paths=['/home/maria/Documents/RainbowMouseCache/facebook_dino-vitb16_embeddings.pkl', '/home/maria/Documents/RainbowMouseCache/google_vit-base-patch16-224_embeddings.pkl',
            '/home/maria/Documents/RainbowMouseCache/openai_clip-vit-base-patch16_embeddings.pkl']

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)['natural_scenes']
    return embeddings

def pca_embs(embeddings, n_components=16):
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    transformed_embeddings = pca.transform(embeddings)
    return transformed_embeddings

embs_matrix=[]
for path in data_paths:
    embeddings = load_embeddings(path)
    transformed_embeddings = pca_embs(embeddings)
    embs_matrix.append(transformed_embeddings)

embs_matrix = np.concatenate(embs_matrix, axis=1)
print(embs_matrix.shape)

np.save('vit_embeddings.npy', embs_matrix)
