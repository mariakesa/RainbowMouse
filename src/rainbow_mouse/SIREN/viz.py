import numpy as np

lfp_data=np.load('lfp_signal.npy')

from rainbow_mouse.SIREN.lib import visualize_embedding_3d_multifreq_siren, animate_embedding_3d_multifreq_siren

#visualize_embedding_3d_multifreq_siren(lfp_data[:3000], "/home/maria/RainbowMouse/src/rainbow_mouse/SIREN/cebra_multifreq_siren.pt", 95, 100, 3, num_layers=3)
animate_embedding_3d_multifreq_siren(
    data=lfp_data[:3000],
    encoder_path="cebra_multifreq_siren.pt",
    input_dim=95,
    hidden_dim=100,
    output_dim=3,
    frequencies=[5, 10, 30, 60],
    num_layers=3,
    save_path="embedding_rotation.mp4"  # or "embedding_rotation.gif"
)