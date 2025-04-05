# umap_html_grid.py
import numpy as np
import pandas as pd
import umap
import os

# Load data
X = np.load('lfp_waveforms.npy')[:1000]
y = np.load('lfp_labels.npy')[:1000]

# Load base64 waveform image strings
def load_waveform_img(idx):
    with open(f"waveform_imgs_base64/img_{idx}.txt", "r") as f:
        return f.read()  # no need for full HTML tag here

waveform_imgs = [load_waveform_img(i) for i in range(len(X))]

# UMAP embedding
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

# Save HTML table with embedded images
df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
df["label"] = y
df["img"] = waveform_imgs

with open("lfp_umap_grid.html", "w") as f:
    f.write("<html><head><title>LFP UMAP Grid</title></head><body>\n")
    f.write("<h2>LFP UMAP Embedding with Waveform Thumbnails</h2>\n")
    f.write("<style>img { height: 40px; }</style>\n")
    f.write("<table border='1' cellspacing='0'>\n")
    f.write("<tr><th>UMAP1</th><th>UMAP2</th><th>Label</th><th>Waveform</th></tr>\n")

    for i, row in df.iterrows():
        f.write(f"<tr><td>{row['UMAP1']:.2f}</td><td>{row['UMAP2']:.2f}</td><td>{row['label']}</td>")
        f.write(f"<td><img src='data:image/png;base64,{row['img']}'></td></tr>\n")

    f.write("</table></body></html>\n")

print("✅ Saved: lfp_umap_grid.html")
