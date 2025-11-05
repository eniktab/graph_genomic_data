
import argparse, json, sys
import numpy as np
from pathlib import Path
from .embedding import WindowEmbedder, EmbeddingSpec
from .cluster import IslandClusterer
from .island import WindowRecord

def cmd_embed(args):
    spec = EmbeddingSpec(window_len=args.window, stride=args.stride, k=args.k, sketch_size=args.sketch)
    we = WindowEmbedder(spec)
    out = []
    for name, start, end, emb in we.embed_fasta(args.fasta):
        out.append({
            "genome": name, "start": start, "end": end,
            "emb": emb.tolist()
        })
    Path(args.out).write_text(json.dumps(out))
    print(f"Wrote {len(out)} window embeddings → {args.out}")

def cmd_cluster(args):
    data = json.loads(Path(args.embeddings).read_text())
    windows = [WindowRecord(d["genome"], d["start"], d["end"], np.asarray(d["emb"], np.float32)) for d in data]
    ic = IslandClusterer(tau=args.tau, min_windows=args.min_windows)
    islands = ic.cluster(windows)
    # Save as a simple JSON summary
    serial = []
    for isl in islands:
        serial.append({
            "island_id": isl.island_id.hex(),
            "centroid_dim": len(isl.centroid),
            "windows": [{"genome":w.genome, "start":w.start, "end":w.end} for w in isl.windows]
        })
    Path(args.out).write_text(json.dumps(serial))
    print(f"Formed {len(islands)} islands → {args.out}")

def main(argv=None):
    p = argparse.ArgumentParser(prog="quilt-phase1")
    sp = p.add_subparsers(dest="cmd", required=True)

    e = sp.add_parser("embed", help="Compute window embeddings (RC-invariant)")
    e.add_argument("fasta")
    e.add_argument("--window", type=int, default=4000)
    e.add_argument("--stride", type=int, default=1000)
    e.add_argument("--k", type=int, default=7)
    e.add_argument("--sketch", type=int, default=256)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_embed)

    c = sp.add_parser("cluster", help="Cluster embeddings into islands by cosine threshold")
    c.add_argument("embeddings", help="JSON from 'embed'")
    c.add_argument("--tau", type=float, default=0.88)
    c.add_argument("--min-windows", type=int, default=1)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_cluster)

    args = p.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
