import pysam

vcf_input = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/Phased_sample.vcf"
output_prefix = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto"

vcf_in = pysam.VariantFile(vcf_input)

samples = list(vcf_in.header.samples)

rec = next(vcf_in)

pos = rec.pos
ref = rec.ref
alts = rec.alts

# Build/collect node for REF allele
ref_id = f"{pos}:{ref}"


# Build/collect nodes for ALT alleles
alt_ids = []
for alt in alts:
    alt_id = f"{pos}:{alt}"
    alt_ids.append(alt_id)

# Gather the chosen node(s) for each sample/haplotype
sample_paths = {}
for sample in samples:
    sample= 'NA00003'
    call = rec.samples[sample]
    gt = call.get("GT")
    print(gt)
    if gt is None:
        continue

    ps = call.get("PS")  # phase-set; may be None/absent
    if ps is None:
        ps = "D"  # TODO fix to work with appropriate phasing

    if sample not in sample_paths:
        sample_paths[sample] = {}
    if ps not in sample_paths[sample]:
        sample_paths[sample][ps] = [[], []]  # diploid => hap0, hap1

    # Convert genotype into node IDs
    for haplotype_idx, allele_idx in enumerate(gt):
        if allele_idx is None or allele_idx < 0:
            continue
        if allele_idx == 0:
            node_id = ref_id
        else:
            # alt index is allele_idx - 1
            if 1 <= allele_idx <= len(alt_ids):
                node_id = alt_ids[allele_idx - 1]
            else:
                continue
        sample_paths[sample][ps][haplotype_idx].append(node_id)

