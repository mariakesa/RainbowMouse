from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm  # for progress bar
from sklearn.preprocessing import StandardScaler

load_dotenv()

# Load paths
allen_path = os.environ.get('ALLEN_CACHE')
rainbow_path = os.environ.get('RAINBOW_MOUSE_CACHE')
manifest_path = os.path.join(allen_path, "manifest.json")

# Load cache and session
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()
session = cache.get_session_data(sessions.index.values[0])

# Get natural scenes, remove blank (-1)
stimuli = session.stimulus_presentations
natural_scenes = stimuli[(stimuli.stimulus_name == "natural_scenes")]

# Get probe E
probe_id = session.probes[session.probes.description == 'probeE'].index.values[0]
lfp = session.get_lfp(probe_id)

# Get LFP timestamps and values
lfp_timestamps = lfp.time[1000:2000]  # shape (T,)
lfp_data = lfp.data # shape (95, T)
lfp_data = lfp.sel(time=slice(100,125))
lfp_data=np.array(lfp_data)
print(f"lfp_data shape: {lfp_data.shape}")
from rainbow_mouse.SIREN.lib import train_cebra_time_multifreq_siren
train_cebra_time_multifreq_siren(lfp_data, 95, 100, 3, num_layers=3, epochs=1000, batch_size=128, learning_rate=1e-3)



