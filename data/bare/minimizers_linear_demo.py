#!/usr/bin/env python3
# Minimizers & Seeding: A Linear, Hands-On Walkthrough (No Functions)
# ---------------------------------------------------------------
# This script mirrors the notebook flow in a single linear file:
# - canonical k-mers
# - minimizer selection via winnowing
# - seed hits between reference and read
# - candidate offset voting from seed diagonals
# Repeats for scenarios: A) exact, B) repetitive, C) SNP, D) reverse-complement, E) too-short
# Plus a small playground at the end.

from collections import defaultdict, Counter

print("# Minimizers & Seeding: Linear Demo (No Functions)\\n")

# Global helpers (not functions)
dna_comp = str.maketrans("ACGTacgt", "TGCAtgca")

# -------------------------
# Utility print banner
# -------------------------
def banner(title):
    print("\\n" + "-"*70)
    print(title)
    print("-"*70)

# =========================
# Scenario A) Exact match
# =========================
banner("A) Exact match")
ref = "ACGTACGT"
read = "GTAC"
k = 3
w = 2
print("Reference:", ref)
print("Read     :", read)
print("k, w     :", k, w)

# Reference k-mers (canonical)
ref_kmers = []
for i in range(len(ref) - k + 1):
    mer = ref[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    if rc < mer:
        ref_kmers.append((rc, i, True))
    else:
        ref_kmers.append((mer, i, False))
print("ref_kmers:", ref_kmers)

# Reference minimizers by winnowing
ref_minimizers = []
if len(ref_kmers) >= w:
    for start in range(0, len(ref_kmers) - w + 1):
        window = ref_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))  # leftmost tie
        if not ref_minimizers or m != ref_minimizers[-1]:
            ref_minimizers.append(m)
print("ref_minimizers:", ref_minimizers)

# Read k-mers (canonical)
read_kmers = []
for i in range(len(read) - k + 1):
    mer = read[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    if rc < mer:
        read_kmers.append((rc, i, True))
    else:
        read_kmers.append((mer, i, False))
print("read_kmers:", read_kmers)

# Read minimizers
read_minimizers = []
if len(read_kmers) >= w:
    for start in range(0, len(read_kmers) - w + 1):
        window = read_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not read_minimizers or m != read_minimizers[-1]:
            read_minimizers.append(m)
print("read_minimizers:", read_minimizers)

# Seed hits: index ref minimizers, join with read minimizers
ref_index = defaultdict(list)  # kmer -> list of (ref_pos, is_revcomp_flag)
for kmer, pos, isrc in ref_minimizers:
    ref_index[kmer].append((pos, isrc))

seed_hits = []
for kmer, rpos, r_isrc in read_minimizers:
    for refpos, _ in ref_index.get(kmer, []):
        seed_hits.append((refpos, rpos, kmer))
print("seed_hits:", seed_hits)

# Candidate offsets from diagonals
offsets = [rp - qp for rp, qp, _ in seed_hits]
offset_counts = Counter(offsets)
print("candidate_offsets:", offset_counts.most_common())
if offset_counts:
    best_d, support = offset_counts.most_common(1)[0]
    print(f"Top candidate offset d = {best_d} (support {support}) -> start near ref[{best_d}]")
else:
    print("No candidate offset.")

# =========================
# Scenario B) Repetitive / Low complexity
# =========================
banner("B) Repetitive / low complexity")
ref = "AAAAAACAAAAAA"
read = "AAAAA"
k = 3
w = 2
print("Reference:", ref)
print("Read     :", read)
print("k, w     :", k, w)

ref_kmers = []
for i in range(len(ref) - k + 1):
    mer = ref[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    ref_kmers.append((rc, i, True) if rc < mer else (mer, i, False))
print("ref_kmers:", ref_kmers)

ref_minimizers = []
if len(ref_kmers) >= w:
    for start in range(0, len(ref_kmers) - w + 1):
        window = ref_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not ref_minimizers or m != ref_minimizers[-1]:
            ref_minimizers.append(m)
print("ref_minimizers:", ref_minimizers)

read_kmers = []
for i in range(len(read) - k + 1):
    mer = read[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    read_kmers.append((rc, i, True) if rc < mer else (mer, i, False))
print("read_kmers:", read_kmers)

read_minimizers = []
if len(read_kmers) >= w:
    for start in range(0, len(read_kmers) - w + 1):
        window = read_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not read_minimizers or m != read_minimizers[-1]:
            read_minimizers.append(m)
print("read_minimizers:", read_minimizers)

ref_index = defaultdict(list)
for kmer, pos, isrc in ref_minimizers:
    ref_index[kmer].append((pos, isrc))

seed_hits = [(rp, qp, m) for m, qp, _ in read_minimizers for rp,_ in ref_index.get(m, [])]
print("seed_hits:", seed_hits)

offsets = [rp - qp for rp, qp, _ in seed_hits]
offset_counts = Counter(offsets)
print("candidate_offsets:", offset_counts.most_common())
if offset_counts:
    best_d, support = offset_counts.most_common(1)[0]
    print(f"Top candidate offset d = {best_d} (support {support})")
else:
    print("No candidate offset.")

# =========================
# Scenario C) SNP breaks seeds
# =========================
banner("C) SNP in the read")
ref = "ACGTACGT"
read = "GTTC"   # differs by one base from GTAC
k = 3
w = 2
print("Reference:", ref)
print("Read     :", read)
print("k, w     :", k, w)

ref_kmers = []
for i in range(len(ref) - k + 1):
    mer = ref[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    ref_kmers.append((rc, i, True) if rc < mer else (mer, i, False))
print("ref_kmers:", ref_kmers)

ref_minimizers = []
if len(ref_kmers) >= w:
    for start in range(0, len(ref_kmers) - w + 1):
        window = ref_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not ref_minimizers or m != ref_minimizers[-1]:
            ref_minimizers.append(m)
print("ref_minimizers:", ref_minimizers)

read_kmers = []
for i in range(len(read) - k + 1):
    mer = read[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    read_kmers.append((rc, i, True) if rc < mer else (mer, i, False))
print("read_kmers:", read_kmers)

read_minimizers = []
if len(read_kmers) >= w:
    for start in range(0, len(read_kmers) - w + 1):
        window = read_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not read_minimizers or m != read_minimizers[-1]:
            read_minimizers.append(m)
print("read_minimizers:", read_minimizers)

ref_index = defaultdict(list)
for kmer, pos, isrc in ref_minimizers:
    ref_index[kmer].append((pos, isrc))

seed_hits = [(rp, qp, m) for m, qp, _ in read_minimizers for rp,_ in ref_index.get(m, [])]
print("seed_hits:", seed_hits)

offsets = [rp - qp for rp, qp, _ in seed_hits]
offset_counts = Counter(offsets)
print("candidate_offsets:", offset_counts.most_common())
if not offset_counts:
    print("No candidate offset: seeds broken by SNP with current k,w. Try smaller k or different w.")

# =========================
# Scenario D) Reverse complement
# =========================
banner("D) Reverse complement (canonical k-mers)")
ref = "ACGTACGT"
# reverse-complement of GTAC
read = "GTAC"[::-1].translate(dna_comp)
k = 3
w = 2
print("Reference:", ref)
print("Read     :", read, "(RC of GTAC)")
print("k, w     :", k, w)

ref_kmers = []
for i in range(len(ref) - k + 1):
    mer = ref[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    ref_kmers.append((rc, i, True) if rc < mer else (mer, i, False))

ref_minimizers = []
if len(ref_kmers) >= w:
    for start in range(0, len(ref_kmers) - w + 1):
        window = ref_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not ref_minimizers or m != ref_minimizers[-1]:
            ref_minimizers.append(m)

read_kmers = []
for i in range(len(read) - k + 1):
    mer = read[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    read_kmers.append((rc, i, True) if rc < mer else (mer, i, False))

read_minimizers = []
if len(read_kmers) >= w:
    for start in range(0, len(read_kmers) - w + 1):
        window = read_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not read_minimizers or m != read_minimizers[-1]:
            read_minimizers.append(m)

ref_index = defaultdict(list)
for kmer, pos, isrc in ref_minimizers:
    ref_index[kmer].append((pos, isrc))

seed_hits = [(rp, qp, m) for m, qp, _ in read_minimizers for rp,_ in ref_index.get(m, [])]
print("seed_hits:", seed_hits)

offsets = [rp - qp for rp, qp, _ in seed_hits]
offset_counts = Counter(offsets)
print("candidate_offsets:", offset_counts.most_common())
if offset_counts:
    best_d, support = offset_counts.most_common(1)[0]
    print(f"Top candidate offset d = {best_d} (support {support})")

# =========================
# Scenario E) Too short for minimizers
# =========================
banner("E) Too short for minimizers")
ref = "ACGTACGT"
read = "AC"
k = 3
w = 2
print("Reference:", ref)
print("Read     :", read)
print("k, w     :", k, w)

ref_kmers = []
for i in range(len(ref) - k + 1):
    mer = ref[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    ref_kmers.append((rc, i, True) if rc < mer else (mer, i, False))

ref_minimizers = []
if len(ref_kmers) >= w:
    for start in range(0, len(ref_kmers) - w + 1):
        window = ref_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not ref_minimizers or m != ref_minimizers[-1]:
            ref_minimizers.append(m)

read_kmers = []
for i in range(len(read) - k + 1):
    mer = read[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    read_kmers.append((rc, i, True) if rc < mer else (mer, i, False))

read_minimizers = []
if len(read_kmers) >= w:
    for start in range(0, len(read_kmers) - w + 1):
        window = read_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not read_minimizers or m != read_minimizers[-1]:
            read_minimizers.append(m)

ref_index = defaultdict(list)
for kmer, pos, isrc in ref_minimizers:
    ref_index[kmer].append((pos, isrc))

seed_hits = [(rp, qp, m) for m, qp, _ in read_minimizers for rp,_ in ref_index.get(m, [])]
print("seed_hits:", seed_hits)

offsets = [rp - qp for rp, qp, _ in seed_hits]
offset_counts = Counter(offsets)
print("candidate_offsets:", offset_counts.most_common())
if not offset_counts:
    print("No candidate offset: read shorter than k+w-1 produced no minimizers.")

# =========================
# Playground: edit and re-run
# =========================
banner("Playground: try your own sequences and parameters")
ref = "GATTACAGATTACA"
read = "TACA"
k = 3
w = 2
print("Reference:", ref)
print("Read     :", read)
print("k, w     :", k, w)

ref_kmers = []
for i in range(len(ref) - k + 1):
    mer = ref[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    ref_kmers.append((rc, i, True) if rc < mer else (mer, i, False))

ref_minimizers = []
if len(ref_kmers) >= w:
    for start in range(0, len(ref_kmers) - w + 1):
        window = ref_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not ref_minimizers or m != ref_minimizers[-1]:
            ref_minimizers.append(m)

read_kmers = []
for i in range(len(read) - k + 1):
    mer = read[i:i+k]
    rc  = mer.translate(dna_comp)[::-1]
    read_kmers.append((rc, i, True) if rc < mer else (mer, i, False))

read_minimizers = []
if len(read_kmers) >= w:
    for start in range(0, len(read_kmers) - w + 1):
        window = read_kmers[start:start+w]
        m = min(window, key=lambda x: (x[0], x[1]))
        if not read_minimizers or m != read_minimizers[-1]:
            read_minimizers.append(m)

ref_index = defaultdict(list)
for kmer, pos, isrc in ref_minimizers:
    ref_index[kmer].append((pos, isrc))

seed_hits = [(rp, qp, m) for m, qp, _ in read_minimizers for rp,_ in ref_index.get(m, [])]
print("seed_hits:", seed_hits)

offsets = [rp - qp for rp, qp, _ in seed_hits]
offset_counts = Counter(offsets)
print("candidate_offsets:", offset_counts.most_common())
if offset_counts:
    best_d, support = offset_counts.most_common(1)[0]
    print(f"Top candidate offset d = {best_d} (support {support})")
print("\\nDone.")
