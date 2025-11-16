# ===============================================================
# Example wiring (kept minimal; adapt to your environment)
# ===============================================================
if __name__ == "__main__":
    import sys, importlib
    def deep_reload(pkg):
        """Reload pkg and all its submodules, leaves -> root."""
        prefix = pkg.__name__ + "."
        mods = [m for m in sys.modules if m == pkg.__name__ or m.startswith(prefix)]
        # deeper modules first (more dots)
        for name in sorted(mods, key=lambda s: s.count("."), reverse=True):
            importlib.reload(sys.modules[name])
    # usage
    import src  # your top-level package
    deep_reload(src)
    from pathlib import Path
    import pysam
    from src.HELIndexer import HELIndexer
    from src.configs import IndexConfig
    from src.ReadAssembler import ReadAssembler, QueryConfig, revcomp
    import logging
    import os, time
    from datetime import datetime
    import random

    WINDOW = 10_000

    N_TILES = 100
    TARGET_LEN = WINDOW * N_TILES


    work = Path("/g/data/te53/en9803/sandpit/graph_genomics/chr22")
    work.mkdir(parents=True, exist_ok=True)
    ref_fa = work / "chm13v2_chr22.fa.gz"          # this should already exist
    tiny_fa = work / "tiny.fa"

    fasta = pysam.FastaFile(str(ref_fa))
    if "chr22" in fasta.references:
        seq = fasta.fetch("chr22")
        out_header = ">chrA\n"
        fasta.close()
        chrom22_seq_clean = seq.upper().replace("N", "")[:TARGET_LEN]
        if not chrom22_seq_clean:
            raise RuntimeError("chr22 sequence is empty after N removal. Check input FASTA?")

        # Write tiny.fa with 60-column lines
        gz = tiny_fa.with_suffix(tiny_fa.suffix + ".gz")

        with tiny_fa.open("wt", newline="\n") as out:
            out.write(">chrA\n")
            for i in range(0, len(chrom22_seq_clean), 60):
                out.write(chrom22_seq_clean[i:i + 60] + "\n")

        # IMPORTANT: compress *after* the writer is closed
        pysam.tabix_compress(str(tiny_fa), str(gz), force=True)

        # Index the BGZF-compressed FASTA
        pysam.faidx(str(gz))

        # Sanity check
        with pysam.FastaFile(str(gz)) as fa:
            print("refs:", fa.references[:3], "…")
            print("chrA length:", fa.get_reference_length("chrA"))

        print(
            f"wrote {tiny_fa} with {len(chrom22_seq_clean)} bp from chr22 (no Ns); "
            f"BGZF={gz.name}; source={ref_fa.name}"
        )




    def make_reads(seq: str, window: int, n: int = 25, seed: int | None = None):
        """
        Generate n reads of length `window` from `seq`.

        - Half of the reads (floor(n/2)) are reverse-complements.
        - Starts are a mix of window-aligned and randomly misaligned ("chopped"):
            * ~50% aligned to multiples of `window`
            * ~50% misaligned anywhere in [0, len(seq)-window]
        - Returns:
            (reads: list[str], forward_starts: list[int])
          where `forward_starts` are the 0-based start indices for the forward (non-RC) reads.
        """
        if window <= 0 or window > len(seq):
            raise ValueError("`window` must be in [1, len(seq)].")

        rng = random.Random(seed) if seed is not None else random

        L = len(seq)
        max_start = L - window
        if max_start < 0:
            raise ValueError("Sequence shorter than window.")

        # Split into aligned vs misaligned starts (keep total == n)
        n_aligned = n // 2
        n_misaligned = n - n_aligned

        # ----- Aligned starts (multiples of `window`) -----
        num_bins = max(1, L // window)  # number of full windows that fit
        aligned_bins = list(range(num_bins))
        if n_aligned <= len(aligned_bins):
            chosen_bins = rng.sample(aligned_bins, n_aligned)
        else:
            # Allow repeats if more aligned reads are requested than bins
            chosen_bins = [aligned_bins[i % len(aligned_bins)] for i in range(n_aligned)]
            rng.shuffle(chosen_bins)

        aligned_starts = [b * window for b in chosen_bins if (b * window) <= max_start]

        # If we lost some due to boundary, top up from remaining aligned bins or clamp
        while len(aligned_starts) < n_aligned:
            s = rng.choice(aligned_bins) * window
            if s <= max_start:
                aligned_starts.append(s)

        # ----- Misaligned ("chopped") starts -----
        domain = range(0, max_start + 1)
        if n_misaligned <= (max_start + 1):
            misaligned_starts = rng.sample(domain, n_misaligned)
        else:
            # Allow repeats if requesting more than available unique positions
            misaligned_starts = [rng.randrange(0, max_start + 1) for _ in range(n_misaligned)]

        # Combine and shuffle for random RC assignment
        starts_all = aligned_starts + misaligned_starts
        rng.shuffle(starts_all)

        # Choose which reads to reverse-complement: exactly floor(n/2)
        k_rc = n // 2
        rc_indices = set(rng.sample(range(len(starts_all)), k_rc))

        reads = []
        forward_starts = []
        for i, s in enumerate(starts_all):
            read = seq[s:s + window]
            if i in rc_indices:
                reads.append(revcomp(read))
            else:
                reads.append(read)
                forward_starts.append(s)

        return reads, forward_starts




    #tiny_fa =  ref_fa

    # ---- timing start ----
    t0 = time.perf_counter()
    print(f"[TIME] start={datetime.now().astimezone().isoformat(timespec='seconds')}")
    # DEBUG is ON by default in QueryConfig; you can still override here
    cfg_query = QueryConfig(
        # Retrieval / chaining: increase recall, reduce early pruning
        top_k=64,  # 32 -> 64
        take_top_chains=8,  # 2 -> 8
        chain_min_hits=1,  # 2 -> 1
        chain_gap_lambda=0.8,  # 1.0 -> 0.8
        competitive_gap=20.0,  # 50.0 -> 20.0

        # Refinement: give WFA/DP more room and run it on more candidates
        flank=0,  # 2000 -> 8000
        refine_top_chains=16,  # 5 -> 16

        # WFA scoring stays as-is unless you have a reason to tweak
        wfa_x=2, wfa_o=3, wfa_e=1,
        wfa_batch_size=256,  # enable efficient batches if GPU path exists

        # DP guards (already generous for 10kb reads)
        dp_max_q_gap=5 * WINDOW,
        dp_diag_band=2 * WINDOW,

        ann_scores_higher_better=True,
        debug_level=logging.WARNING,
        use_gpu_for_numpy_hits_threshold=50_000,
    )

    cfg_index = IndexConfig(window=WINDOW, stride=5000, rc_index=True)
    hel = HELIndexer(tiny_fa, cfg_index, emb_batch=1024)
    hel.build_or_load(tiny_fa.parent / "index", reuse_existing=False, verbose=True)
    assembler = ReadAssembler(hel_indexer=hel, cfg=cfg_query, window=WINDOW, stride=None)

    reads = [revcomp(chrom22_seq_clean[i:i+WINDOW]) for i in range(0, WINDOW*5, WINDOW)]
    placements = assembler.assemble(reads)
    for p in placements:
        print(p)


   # reads, starts = make_reads(chrom22_seq_clean, WINDOW , n=10, seed=None)

    #print(starts)
    #placements = assembler.assemble(reads)
    #for p in placements:
    #    print(p)


    # ---- timing end ----
    t1 = time.perf_counter()
    print(f"[TIME] end={datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"[TIME] elapsed_sec={t1 - t0:.3f}")