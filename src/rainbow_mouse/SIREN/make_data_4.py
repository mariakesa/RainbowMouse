import numpy as np
from rainbow_mouse.SIREN.lib import train_cebra_time_learnable_siren

data=np.load('/home/maria/RainbowMouse/src/rainbow_mouse/SIREN/lfp_signal.npy')
input_dim=95
hidden_dim=100
output_dim=3
train_cebra_time_learnable_siren(
    data,
    input_dim,
    hidden_dim,
    output_dim,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    epochs=1000,
    batch_size=128,
    learning_rate=1e-3,
    model_path="cebra_learnable_siren.pt"
)