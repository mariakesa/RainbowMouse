from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm  # for progress bar

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
natural_scenes = stimuli[(stimuli.stimulus_name == "natural_scenes") & (stimuli.frame != -1)]

# Get probe E
probe_id = session.probes[session.probes.description == 'probeE'].index.values[0]
lfp = session.get_lfp(probe_id)

# Get LFP timestamps and values
lfp_timestamps = lfp.time  # shape (T,)
lfp_data = lfp.data.T  # shape (95, T)

# Build data matrix
n_channels = lfp_data.shape[0]
n_trials = natural_scenes.shape[0]

X = np.zeros((n_channels, n_trials))
y = np.zeros(n_trials, dtype=int)

print("Averaging LFP values for each stimulus trial...")

for i, (_, row) in tqdm(enumerate(natural_scenes.iterrows()), total=n_trials):
    start_time = row.start_time
    stop_time = row.stop_time
    frame = int(row.frame)

    # Find LFP timestamps within trial window
    mask = (lfp_timestamps >= start_time) & (lfp_timestamps <= stop_time)
    trial_lfp = lfp_data[:, mask]

    if trial_lfp.shape[1] == 0:
        print(f"Warning: No LFP data for trial {i} between {start_time}-{stop_time}")
        X[:, i] = np.nan
    else:
        X[:, i] = trial_lfp.mean(axis=1)

    y[i] = frame

print(X.shape)
print(y.shape)
# Save if needed
np.save("lfp_X.npy", X)
np.save("lfp_y.npy", y)

