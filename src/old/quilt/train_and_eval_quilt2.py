#!/usr/bin/env python3

"""
Usage:
  python train_and_eval_quilt2.py --ref path/to/reference.fasta --reads path/to/reads.fasta \
      --win 4000 --stride 1000 --epochs 3 --batch 64 --d 256 --out_dir ./out

What it does:
  1) Loads a reference FASTA, chops into fixed windows (win/stride).
  2) Trains the RC-invariant read embedder (TensorFlow/Keras) with NT-Xent contrastive loss on windows.
  3) Builds island embeddings (one island per reference window) and a GPU cosine index.
  4) Assigns each read (FASTA) to the most similar island (window) using the trained model.
"""
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

# --- Import the Phase-2 package (expect quilt2 directory in PYTHONPATH) ---
from src.old.quilt.tokenize import one_hot_dna
from src.old.quilt.model import build_read_embedder_base
from src.old.quilt.assign import IslandAssigner
from src.old.quilt.fasta import read_fasta
from src.old.quilt.layers import RCInvariant

def window_reference(ref_records, win=4000, stride=1000):
    out=[]
    for name, seq in ref_records:
        L=len(seq)
        if L<=win:
            out.append((name, 0, L, seq))
            continue
        for s in range(0, max(1, L-win+1), stride):
            e = min(L, s+win)
            out.append((name, s, e, seq[s:e]))
    return out

def augment_seq_fixedlen(seq: str, target_len: int, sub_rate=0.01, ins_rate=0.002, del_rate=0.002, jitter=32):
    bases = ["A","C","G","T"]
    start_jit = random.randint(-jitter, jitter)
    start = max(0, min(len(seq)-1, 0 + start_jit))
    s = list(seq[start: start + target_len + 64])
    i=0; out=[]
    while i < len(s) and len(out) < target_len + 64:
        b = s[i]
        if random.random() < del_rate:
            i += 1
            continue
        if random.random() < sub_rate and b in "ACGT":
            out.append(random.choice([x for x in bases if x!=b]))
        else:
            out.append(b)
        if random.random() < ins_rate:
            out.append(random.choice(bases))
        i += 1
    out = out[:target_len]
    if len(out) < target_len:
        out += ["N"]*(target_len - len(out))
    return "".join(out)

def rc(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]

class NTXent(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name="NTXent"):
        super().__init__(name=name)
        self.t = temperature
    def call(self, z1, z2):
        z1 = tf.math.l2_normalize(z1, axis=-1)
        z2 = tf.math.l2_normalize(z2, axis=-1)
        B = tf.shape(z1)[0]
        Z = tf.concat([z1, z2], axis=0)
        sim = tf.linalg.matmul(Z, Z, transpose_b=True)
        large_neg = -1e9
        mask = tf.eye(2*B, dtype=tf.bool)
        sim = tf.where(mask, large_neg*tf.ones_like(sim), sim)
        pos = tf.concat([tf.range(B, 2*B), tf.range(0, B)], axis=0)
        pos_sim = tf.gather(sim, pos, axis=1, batch_dims=0)
        logits = sim / self.t
        pos_logits = pos_sim / self.t
        logsumexp = tf.reduce_logsumexp(logits, axis=1)
        loss = logsumexp - tf.squeeze(pos_logits, axis=-1)
        return tf.reduce_mean(loss)

def make_dataset_from_windows(windows, batch_size=64, win=4000):
    seqs = [w[3] for w in windows]
    L = win
    def gen():
        while True:
            idx = np.random.randint(0, len(seqs), size=(batch_size,))
            for i in idx:
                s = seqs[i]
                a1 = augment_seq_fixedlen(s, L)
                a2 = augment_seq_fixedlen(s, L)
                if random.random() < 0.5: a1 = rc(a1)
                if random.random() < 0.5: a2 = rc(a2)
                yield (a1, a2)
    def to_one_hot(a, b):
        oh1 = one_hot_dna(a)
        oh2 = one_hot_dna(b)
        return (oh1, oh2)
    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    ))
    ds = ds.map(lambda a,b: to_one_hot(a,b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train_embedder(windows, win=4000, d_model=256, epochs=3, batch=64, steps_per_epoch=200, lr=1e-3, reduce="mean"):
    ds = make_dataset_from_windows(windows, batch_size=batch, win=win)
    base = build_read_embedder_base(seq_len=win, d_model=d_model, channels=5)
    opt = tf.keras.optimizers.Adam(lr)
    loss_fn = NTXent(temperature=0.1)
    @tf.function
    def train_step(oh1, oh2):
        with tf.GradientTape() as tape:
            z1 = base(oh1, training=True)
            z2 = base(oh2, training=True)
            loss = loss_fn(z1, z2)
        grads = tape.gradient(loss, base.trainable_variables)
        opt.apply_gradients(zip(grads, base.trainable_variables))
        return loss
    for ep in range(1, epochs+1):
        avg = tf.keras.metrics.Mean()
        for step, (oh1, oh2) in enumerate(ds.take(steps_per_epoch), start=1):
            loss = train_step(oh1, oh2)
            avg.update_state(loss)
            if step % 50 == 0:
                print(f"Epoch {ep} step {step}/{steps_per_epoch} - loss {avg.result().numpy():.4f}")
        print(f"Epoch {ep} done. mean loss={avg.result().numpy():.4f}")
    inp = tf.keras.Input(shape=(win,5))
    emb = RCInvariant(base, reduce=reduce)(inp)
    model = tf.keras.Model(inp, emb, name="read_embedder_rc_trained")
    return model

def build_island_index(model, windows, win=4000):
    ids = []
    embs = []
    for i, (name, s, e, seq) in enumerate(windows):
        oh = one_hot_dna(seq)
        z = model(oh[None,...], training=False).numpy()[0]
        z = z / (np.linalg.norm(z)+1e-12)
        ids.append(f"{name}:{s}-{e}".encode())
        embs.append(z.astype("float32"))
    embs = np.vstack(embs)
    return ids, embs

def assign_reads(model, island_ids, island_embs, reads_records, win=4000, stride=1000, top_k=5, windowed=True):
    assigner = IslandAssigner(island_ids, island_embs, d_model=island_embs.shape[1], seq_len=win)
    results = []
    for name, seq in reads_records:
        res = assigner.assign_read(seq, top_k=top_k, windowed=windowed, win=win, stride=stride)
        results.append((name, res))
    return results


ref = "/home/niktabel/workspace/sync/ANU/graph_genomic_data/data/bare/reference.fa"
reads", required=True, help="Reads FASTA")
win", type=int, default=4000)
stride", type=int, default=1000)
epochs", type=int, default=3)
steps", type=int, default=200)
batch", type=int, default=64)
d", type=int, default=256)
lr", type=float, default=1e-3)
out_dir", type=str, default="./out")


out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
print("Loading reference...")
ref = read_fasta(args.ref)
windows = window_reference(ref, win=args.win, stride=args.stride)
if len(windows) < 2:
    raise SystemExit("Need at least 2 windows from reference for training. Reduce win or stride.")
print(f"Reference windows: {len(windows)}")

print("Training RC-invariant embedder (contrastive)...")
model = train_embedder(windows, win=args.win, d_model=args.d, epochs=args.epochs, batch=args.batch, steps_per_epoch=args.steps, lr=args.lr)
model.save(out / "rc_embedder_savedmodel")

print("Building island index from reference windows...")
island_ids, island_embs = build_island_index(model, windows, win=args.win)
np.save(out / "island_ids.npy", np.array(island_ids, dtype=object))
np.save(out / "island_embs.npy", island_embs)

print("Loading reads...")
reads = read_fasta(args.reads)
print(f"Reads: {len(reads)}")

print("Assigning reads to islands...")
assignments = assign_reads(model, island_ids, island_embs, reads, win=args.win, stride=args.stride, top_k=5, windowed=True)

report_path = out / "assignments.tsv"
with open(report_path, "w") as f:
    print("read_name\ttype\tbest_island\tscore\textra", file=f)
    for name, res in assignments:
        if res["type"] in ("too_short","too_noisy"):
            print(f"{name}\t{res['type']}\t.\t.\t{res}", file=f)
        elif res["type"] == "windowed":
            con = res["consensus"]
            print(f"{name}\twindowed\t{con['island_id']}\tNA\tsegments={len(res['segments'])}", file=f)
        else:
            best = res["best"]
            print(f"{name}\t{res['type']}\t{best['island_id']}\t{best['sim']:.3f}\t.", file=f)

print(f"Done. Report: {report_path}")

