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

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def visualize_lfp_umap(lfp, stimulus_df, channel_indices):
    """
    Extracts LFP waveforms aligned to stimulus presentations and visualizes them using UMAP.
    
    Parameters:
    - lfp: LFP object from AllenSDK.
    - stimulus_df: DataFrame of stimulus presentations with 'start_time' and 'stop_time'.
    - channel_indices: List or array of channel indices to use.
    - pre_time: Seconds before stimulus start to include.
    - post_time: Seconds after stimulus start to include.
    - sample_rate: LFP sample rate in Hz (default 2500).
    """
    times = lfp.time
    data = lfp.data[:, channel_indices]  # shape: [time, selected_channels]
    print(data.shape)
    waveforms = []

    lens=[]
    labels = []
    for _, row in tqdm(stimulus_df.iterrows(), total=len(stimulus_df)):
        start = row.start_time 
        stop = row.stop_time
        labels.append(row.frame)

        # Find time indices
        idx_start = np.searchsorted(times, start)
        idx_stop = np.searchsorted(times, stop)
        
        if idx_stop > data.shape[0] or idx_start < 0 or idx_stop <= idx_start:
            continue

        segment = data[idx_start:idx_stop, :]  # shape: [time_window, n_channels]

        waveforms.append(segment.flatten())  # Flatten to [time × channels]
        lens.append(len(segment))
    if len(waveforms) == 0:
        print("No waveforms extracted.")
        return
    # Pad waveforms to the same length
    min_length = min(lens)
    print(min_length)
    waveforms=[w[:min_length] for w in waveforms]

    X = np.stack(waveforms)  # shape: [n_trials, time × channels]

    # Normalize each feature across trials
    X = StandardScaler().fit_transform(X)
    labels= np.array(labels)
    print(X.shape)

    np.save("lfp_waveforms.npy", X)
    np.save("lfp_labels.npy", labels)
channel_idxs = [50]  # Use first 10 channels as an example
visualize_lfp_umap(lfp, natural_scenes, channel_idxs)
