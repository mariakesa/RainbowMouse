from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

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
n_trials = len(natural_scenes)

example_probe_id = session.probes[session.probes.description == 'probeE'].index.values[0]
lfp_example = session.get_lfp(example_probe_id)
lfp_timestamps = lfp_example.time  # shape: (T,)

# Prepare output
all_X = []
probe_names = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']

print("Averaging LFP values per trial, per probe (memory efficient)...")

for probe_name in tqdm(probe_names, desc="Processing probes"):
    probe_id = session.probes[session.probes.description == probe_name].index.values[0]
    lfp = session.get_lfp(probe_id)
    lfp_timestamps = lfp.time               # shape: (T,)
    n_channels = lfp.data.shape[1]
    n_timepoints = lfp.data.shape[0]

    print(f"Probe: {probe_name} — timepoints: {n_timepoints}, channels: {n_channels}")
    X_probe = np.zeros((n_channels, n_trials))  # [channels, trials]

    for i, (_, row) in enumerate(natural_scenes.iterrows()):
        start_time = row.start_time
        stop_time = row.stop_time

        start_idx = lfp.time.searchsorted(start_time, side="left")
        stop_idx = lfp.time.searchsorted(stop_time, side="right")

        if stop_idx <= start_idx:
            X_probe[:, i] = np.nan
            continue

        chunk = lfp.data[start_idx:stop_idx, :]  # memory-mapped slice
        X_probe[:, i] = np.mean(chunk, axis=0, dtype=np.float32)

        # Clean up explicitly
        del chunk

    del lfp

    all_X.append(X_probe)



# Final concatenation
X = np.concatenate(all_X, axis=0)  # shape: [total_channels, n_trials]
y = natural_scenes['frame'].astype(int).values  # [n_trials]

# Save
np.save(f"{rainbow_path}/lfp_multi.npy", X)
np.save(f"{rainbow_path}/frames_multi.npy", y)

print("✅ DATA SAVED")
print("LFP shape:", X.shape)
print("Frame labels shape:", y.shape)
