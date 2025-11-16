# ===============================================================
# Example wiring (kept minimal; adapt to your environment)
# ===============================================================
if __name__ == "__main__":
    import sys
    print('Python %s on %s' % (sys.version, sys.platform))
    sys.path.extend(['/g/data/te53/en9803/workspace/sync/ANU/graph_genomic_data/'])
    from pathlib import Path
    from src.HELIndexer import HELIndexer
    from src.configs import IndexConfig
    from src.helpers import GPUMonitor, StageTimer
    import logging
    import datetime as dt
    import time
    import os

    WINDOW = 10_000

    # --------------------------
    # Config (you can tweak)
    # --------------------------
    WINDOW = 10_000
    SAMPLE_INTERVAL_SEC = 1.0  # GPU sampling interval


    tiny_fa = Path("/g/data/te53/en9803/data/Ref/CHM13/GIAB/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta.gz")
    index_dir = tiny_fa.parent / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = index_dir / f"HELIndexer_run_{ts}.log"

    # --------------------------
    # Logging setup
    # --------------------------
    logger = logging.getLogger("HELIndexerRun")
    logger.setLevel(logging.INFO)

    # File handler (full info)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler (only warnings+ to keep console clean)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(ch)

    logger.info("===== HELIndexer run started =====")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    logger.info(f"Input FASTA: {tiny_fa}")
    logger.info(f"Index directory: {index_dir}")

    t_total_start = time.perf_counter()
    # --------------------------
    # Build/Load Index with GPU monitoring
    # --------------------------
    cfg_index = IndexConfig(window=WINDOW, stride=5000, rc_index=True, ids_max_tokens_per_call=262_144)
    hel = HELIndexer(tiny_fa, cfg_index, emb_batch=512)

    gpu_mon = GPUMonitor(interval=SAMPLE_INTERVAL_SEC, logger=logger)
    gpu_mon.start()
    logger.info("GPU monitor started.")

    with StageTimer("HELIndexer.build_or_load", logger):
        hel.build_or_load(index_dir, reuse_existing=False, include_dataset=True, verbose=True)

    # Stop GPU sampling and summarize
    gpu_mon.stop()
    gpu_mon.join(timeout=5.0)
    summary = gpu_mon.summarize()
    gpu_mon.close()

    logger.info("----- GPU Utilization Summary -----")
    for idx, s in summary.items():
        if not s:
            continue


        def fmt(stat: dict):
            if not stat or stat["mean"] is None:
                return "NA"
            return f"mean={stat['mean']:.1f}, min={stat['min']:.1f}, max={stat['max']:.1f}"


        logger.info(
            f"[GPU {idx}] "
            f"util%({fmt(s['util_pct'])}); "
            f"mem_used_MB({fmt(s['mem_used_MB'])}); "
            f"power_W({fmt(s['power_W'])}); "
            f"temp_C({fmt(s['temp_C'])}); "
            f"sm_clock_MHz({fmt(s['sm_clock_MHz'])}); "
            f"mem_clock_MHz({fmt(s['mem_clock_MHz'])}); "
            f"samples={s['samples']}, duration={s['duration_sec']:.1f}s"
        )

    total_dt = time.perf_counter() - t_total_start
    logger.info(f"[TIMER] TOTAL runtime: {total_dt:.3f} s.")
    logger.info("===== HELIndexer run completed =====")
    logger.info(f"Log written to: {log_file}")

