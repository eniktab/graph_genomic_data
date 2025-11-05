import random
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
import torch.nn.functional as F
import math

work = Path("/g/data/te53/en9803//sandpit/graph_genomics/chr22")
work.mkdir(parents=True, exist_ok=True)
ref_fa = work / "chr22.fa"
recs = list(SeqIO.parse(str(ref_fa), "fasta"))
# Grab by ID to be robust to order
chromA = str(next(r.seq for r in recs if r.id == "chr22")).replace("N", "")[:5994 * 10]

cfg = IndexConfig(
    window=5994,
    stride=5994,
    rc_index=True,
    model_name="nucleotide-transformer-2.5b-1000g",
    model_dir="/g/data/te53/en9803/data/scratch/hf-cache/models/nucleotide-transformer-2.5b-1000g",
)

metas: List[Tuple[str, int, int]] = []
seqs: List[str] = []
chrom = "chr22"
s = chromA.upper()
for start, sub in chunk_sequence(s, cfg.window, cfg.stride):
    metas.append((chrom, start, +1));
    seqs.append(sub)
    #metas.append((chrom, start, -1));
    #seqs.append(revcomp(sub))


backend = NTBackend(model_name=cfg.model_name,
                                  model_dir=cfg.model_dir)

emb_chunks = backend.embed_list(seqs)


E = torch.nn.functional.normalize(emb_chunks, p=2, dim=1)
sim_mat = E @ E.T          # [2, 2]
dist_mat = 1.0 - sim_mat
print(dist_mat[0, 1].item())  # same scalar distance



Ap, Bp = embeddings[0].mean(0), embeddings[1].mean(0)
torch.norm(out[0] - out[1], p=2)


F.cosine_similarity(emb_chunks[0], emb_chunks[1], dim=0)
F.cosine_similarity(embeddings[0], embeddings[1], dim=0).mean()




emb_chunks = []
for i in range(0, 2560, 256):
    emb_chunks.append(backend.embed_list(seqs[i:i + 256]))
embs = torch.cat(emb_chunks, dim=0)  # (N, H) on CUDA
dim = embs.shape





emb_chunks = backend.embed_list(seqs)



# emb_chunks: tensor([[...], [...]]) on cuda
a = emb_chunks[0].to(torch.float64)
b = emb_chunks[1].to(torch.float64)



