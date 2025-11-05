
import tensorflow as tf, numpy as np
from .tokenize import one_hot_dna
from .model import build_read_embedder_rc
from .index import DotProductIndex
def sliding_window_segments(seq:str, win:int=4000, stride:int=1000):
    L=len(seq)
    if L<=win: return [(0,L,seq)]
    return [(s, min(L,s+win), seq[s:min(L,s+win)]) for s in range(0, max(1,L-win+1), stride)]
class IslandAssigner:
    def __init__(self, island_ids:list[bytes], island_embeddings: np.ndarray, d_model=256, seq_len=None, pooling="mean"):
        self.index=DotProductIndex.from_numpy(island_embeddings, [bytes(i).hex() if isinstance(i,(bytes,bytearray)) else i for i in island_ids])
        self.model=build_read_embedder_rc(seq_len=seq_len, d_model=d_model, channels=5, reduce=pooling)
    def embed_seq(self, seq:str)->np.ndarray:
        oh=one_hot_dna(seq); return self.model(oh[None,...], training=False).numpy()[0]
    def assign_read(self, seq:str, top_k=5, windowed=True, win=4000, stride=1000, min_len=200, n_frac_thresh=0.5, score_thresh=0.6):
        L=len(seq)
        if L<min_len: return {"type":"too_short","length":L}
        nfrac=seq.count("N")/max(1,L)
        if nfrac>n_frac_thresh: return {"type":"too_noisy","n_frac":nfrac}
        segs=[(0,L,seq)]
        if windowed and L>win: segs=sliding_window_segments(seq, win, stride)
        results=[]
        for s,e,sub in segs:
            oh=one_hot_dna(sub)
            q=self.model(oh[None,...], training=False)
            sims, ids=self.index.search(q, k=top_k)
            sims=sims[0]; ids=[i.decode() for i in ids[0]]
            results.append({"start":s,"end":e,"topk":[{"island_id":ids[j],"sim":float(sims[j])} for j in range(len(ids))]})
        if len(results)==1:
            best=results[0]["topk"][0]
            label="confident" if best["sim"]>=score_thresh else "uncertain"
            return {"type":label, "best":best, "topk":results[0]["topk"]}
        votes={}
        for r in results:
            bid=r["topk"][0]["island_id"]; votes[bid]=votes.get(bid,0)+1
        consensus=max(votes.items(), key=lambda x:x[1])
        return {"type":"windowed","consensus":{"island_id":consensus[0],"votes":int(consensus[1])},"segments":results}
