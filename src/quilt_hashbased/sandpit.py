import json
import numpy as np
from pathlib import Path
from src.old.quilt import WindowEmbedder, EmbeddingSpec
from src.old.quilt import IslandClusterer
from src.old.quilt import WindowRecord

window = 10
stride= 8
k = 3
sketch = 8

fasta = "/home/niktabel/workspace/sync/ANU/graph_genomic_data/data/bare/reference.fa"
emb_out= "/home/niktabel/workspace/sync/ANU/graph_genomic_data/data/sanpit_output/emb.json"
embeddings = "/home/niktabel/workspace/sync/ANU/graph_genomic_data/data/sanpit_output/emb.json"
min_windows = 1
tau = 0.88
island_out = "/home/niktabel/workspace/sync/ANU/graph_genomic_data/data/sanpit_output/island.json"


spec = EmbeddingSpec(window_len=window, stride=stride, k=k, sketch_size=sketch)
we = WindowEmbedder(spec)
out = []
for name, start, end, emb in we.embed_fasta(fasta):
    out.append({
        "genome": name, "start": start, "end": end,
        "emb": emb.tolist()
    })
Path(emb_out).write_text(json.dumps(out))

data = json.loads(Path(embeddings).read_text())
windows = [WindowRecord(d["genome"], d["start"], d["end"], np.asarray(d["emb"], np.float32)) for d in data]
ic = IslandClusterer(tau=tau, min_windows=min_windows)
islands = ic.cluster(windows)
# Save as a simple JSON summary
serial = []
for isl in islands:
    serial.append({
        "island_id": isl.island_id.hex(),
        "centroid_dim": len(isl.centroid),
        "windows": [{"genome": w.genome, "start": w.start, "end": w.end} for w in isl.windows]
    })
Path(island_out).write_text(json.dumps(serial))
print(f"Formed {len(islands)} islands → {out}")