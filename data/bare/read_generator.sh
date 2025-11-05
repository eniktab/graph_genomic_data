\
#!/usr/bin/env bash
set -euo pipefail
eval "$(mamba shell hook --shell bash)"
mamba activate pbsim3

# =========[ TRACE / VERBOSITY ]=========
# TRACE=1 enables shell xtrace (very verbose); default off
TRACE="${TRACE:-0}"
if [[ "$TRACE" == "1" ]]; then set -x; fi

# =========[ PATHS / TOOLS ]=========
PBS_PATH="/home/niktabel/workspace/sandpit/longread/pbsim3"
export PATH="/home/niktabel/workspace/sandpit/longread/art_bin_MountRainier:/home/niktabel/workspace/sandpit/longread/ccs:$PATH"

# ----------------------------
# Config
# ----------------------------
PB_MODEL_QSHMM="$PBS_PATH/data/QSHMM-RSII.model"     # PBSIM3 model for PacBio (quality-score HMM)
ONT_MODEL_QSHMM="$PBS_PATH/data/QSHMM-ONT-HQ.model"  # PBSIM3 model for ONT (HQ) 
HIFI_PASSES=10                              # number of passes for HiFi simulation
PB_COV=20                                   # desired per-haplotype coverage for PacBio (before ccs)
ONT_COV=20                                  # desired per-haplotype coverage for ONT
ILMN_COV=50                                 # desired total diploid coverage for Illumina PE
READ_LEN=150                                 # Illumina read length
FRAG_MEAN=350                                # Illumina insert mean
FRAG_SD=35                                   # Illumina insert SD
THREADS=8

# Input references
HAP_A="hapA.fa"
HAP_B="hapB.fa"

# Output dirs
mkdir -p sim/hifi sim/ont sim/illumina

# ----------------------------------------------------
# 1) PacBio HiFi via PBSIM3 multi-pass  + ccs (HiFi)
#    (simulate subreads -> BAM, then run ccs to HiFi)
# ----------------------------------------------------
# hapA
pbsim \
  --strategy wgs \
  --method qshmm \
  --qshmm "${PB_MODEL_QSHMM}" \
  --depth ${PB_COV} \
  --genome "${HAP_A}" \
  --pass-num ${HIFI_PASSES} \
  --prefix sim/hifi/hapA

# hapB
pbsim \
  --strategy wgs \
  --method qshmm \
  --qshmm "${PB_MODEL_QSHMM}" \
  --depth ${PB_COV} \
  --genome "${HAP_B}" \
  --pass-num ${HIFI_PASSES} \
  --prefix sim/hifi/hapB
  
# Convert subread BAM -> HiFi FASTQ with ccs (aka pbccs)
# (ccs will produce circular consensus sequences from the multi-pass BAM)
ccs ./sim/hifi/hapA_0001.bam sim/hifi/hapA.hifi.fastq.gz --report-json ./sim/hifi/hapA.ccs.report.json
ccs ./sim/hifi/hapB_0001.bam sim/hifi/hapB.hifi.fastq.gz --report-json ./sim/hifi/hapB.ccs.report.json

# Optional: combine HiFi reads for a "diploid" mixture
cat ./sim/hifi/hapA.hifi.fastq.gz sim/hifi/hapB.hifi.fastq.gz > sim/hifi/diploid.hifi.fastq.gz

# ----------------------------------------------------
# 2) ONT long reads (PBSIM3)
# ----------------------------------------------------
# hapA
pbsim \
  --strategy wgs \
  --method qshmm \
  --qshmm "${ONT_MODEL_QSHMM}" \
  --depth ${ONT_COV} \
  --genome "${HAP_A}" \
  --prefix sim/ont/hapA

# hapB
pbsim \
  --strategy wgs \
  --method qshmm \
  --qshmm "${ONT_MODEL_QSHMM}" \
  --depth ${ONT_COV} \
  --genome "${HAP_B}" \
  --prefix sim/ont/hapB

# PBSIM3 ONT outputs FASTQ directly (gz). Combine for mixture:
cat ./sim/ont/hapA_0001.fq.gz sim/ont/hapB_0001.fq.gz > sim/ont/diploid.ont.fastq.gz

