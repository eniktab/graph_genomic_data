import os
import pysam
from concurrent.futures import ProcessPoolExecutor
import src.old.proto.vg_pb2 as vg_pb2  # <-- This should be your VG proto Python module

class VgHaplotypeGraphBuilder:
    """
    Builds a VG-style Graph from a phased VCF (by chromosome).
    Also creates a Path for each (sample, phase_set, hap_idx) haplotype.
    """

    def __init__(self, chrom: str):
        self.chrom = chrom
        self.graph = vg_pb2.Graph()
        self.node_key_to_id = {}
        self.next_id = 1

        # last_node[(sample, ps, hap_idx)] = node_id
        self.last_node = {}

        # (sample, ps, hap_idx) -> vg_pb2.Path
        self.paths = {}

        # Optionally create a special "start" node
        self.start_node_id = self._add_node(sequence="", name="START_NODE")

    def _add_node(self, sequence: str, name: str = "") -> int:
        node_id = self.next_id
        self.next_id += 1

        node = self.graph.node.add()
        node.id = node_id
        node.sequence = sequence
        if name:
            node.name = name
        return node_id

    def _get_or_create_node_id(self, pos: int, allele_str: str) -> int:
        key = (pos, allele_str)
        if key in self.node_key_to_id:
            return self.node_key_to_id[key]

        node_id = self._add_node(
            sequence=allele_str,
            name=f"{pos}:{allele_str}"
        )
        self.node_key_to_id[key] = node_id
        return node_id

    def _add_edge(self, from_id: int, to_id: int) -> None:
        edge = self.graph.edge.add()
        edge.from_ = from_id
        edge.to = to_id
        # By default, from_start/to_end/overlap are False/False/0

    def _get_path(self, sample: str, ps: str, hap_idx: int) -> vg_pb2.Path:
        key = (sample, ps, hap_idx)
        if key not in self.paths:
            path = vg_pb2.Path()
            path.name = f"{sample}_PS{ps}_hap{hap_idx}"
            self.paths[key] = path
        return self.paths[key]

    def _add_mapping_to_path(self, path: vg_pb2.Path, node_id: int, allele_len: int):
        mapping = path.mapping.add()
        # We can choose 1-based or 0-based rank; here we just use the count
        mapping.rank = len(path.mapping)

        mapping.position.node_id = node_id
        mapping.position.offset = 0

        if allele_len > 0:
            edit = mapping.edit.add()
            edit.from_length = allele_len
            edit.to_length = allele_len
            # sequence is empty => perfect match

    def _record_allele_visit(self, sample: str, ps: str, hap_idx: int, node_id: int, allele_str: str):
        key = (sample, ps, hap_idx)
        prev_node_id = self.last_node.get(key)

        if prev_node_id is None:
            self._add_edge(self.start_node_id, node_id)
        else:
            self._add_edge(prev_node_id, node_id)

        self.last_node[key] = node_id

        path = self._get_path(sample, ps, hap_idx)
        self._add_mapping_to_path(path, node_id, len(allele_str))

    def process_record(self, rec, samples) -> None:
        if not rec.alts:
            return

        # Check if any sample has an ALT allele
        any_alt_in_samples = any(
            any(a is not None and a > 0 for a in rec.samples[sample].get("GT", []))
            for sample in samples
        )
        if not any_alt_in_samples:
            return

        pos = rec.pos
        ref_allele = rec.ref
        alt_alleles = rec.alts

        # Node for REF
        ref_node_id = self._get_or_create_node_id(pos, ref_allele)

        # Node(s) for ALT(s)
        alt_node_ids = []
        for alt in alt_alleles:
            alt_node_id = self._get_or_create_node_id(pos, alt)
            alt_node_ids.append(alt_node_id)

        # For each sample, link to the appropriate node
        for sample in samples:
            call = rec.samples[sample]
            gt = call.get("GT") or ()
            if not gt:
                continue

            ps = call.get("PS") or "D"

            for hap_idx, allele_idx in enumerate(gt):
                if allele_idx is None or allele_idx < 0:
                    continue
                if allele_idx == 0:
                    self._record_allele_visit(sample, ps, hap_idx, ref_node_id, ref_allele)
                else:
                    if 1 <= allele_idx <= len(alt_node_ids):
                        alt_node_id = alt_node_ids[allele_idx - 1]
                        alt_str = alt_alleles[allele_idx - 1]
                        self._record_allele_visit(sample, ps, hap_idx, alt_node_id, alt_str)

    def build_graph(self) -> vg_pb2.Graph:
        # Copy paths into the graph
        for path_obj in self.paths.values():
            new_path = self.graph.path.add()
            new_path.CopyFrom(path_obj)
        return self.graph


def build_graph_for_chrom(vcf_path: str, chrom: str) -> bytes:
    """
    Worker function that opens the VCF itself, fetches `chrom`,
    and builds a haplotype graph as VG .proto.
    """
    with pysam.VariantFile(vcf_path) as vcf_in:
        samples = list(vcf_in.header.samples)
        builder = VgHaplotypeGraphBuilder(chrom)

        # If the VCF is indexed, we can do a fetch.
        # If not, consider scanning all records and filtering by rec.chrom == chrom.
        for rec in vcf_in.fetch(chrom):
            builder.process_record(rec, samples)

    vg_graph = builder.build_graph()
    return vg_graph.SerializeToString()


def vcf_to_vg_haplotypes(
    vcf_path: str,
    output_prefix: str,
    contigs: list[str] = None,
    max_workers: int = None
) -> None:
    """
    Creates .vg files for each chromosome (or contig) in the VCF.
    Each worker opens the VCF, processes just that contig,
    and returns serialized bytes with haplotype paths.
    """

    # 1) If user didn't supply a list of contigs, attempt to get them from the header
    if contigs is None:
        with pysam.VariantFile(vcf_path) as vf:
            contigs = list(vf.header.contigs.keys())
            # If still empty, optionally fallback to scanning all records to find unique chrom names
            if not contigs:
                vf.reset()  # ensure we can read from the start
                chrom_set = set()
                for rec in vf:
                    chrom_set.add(rec.chrom)
                contigs = sorted(chrom_set)

    # 2) If we ended up with nothing, exit
    if not contigs:
        print("No contigs (or chromosomes) found! Exiting.")
        return

    # 3) Parallel or serial
    max_workers = max_workers or os.cpu_count() or 1
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chrom in contigs:
            futures[chrom] = executor.submit(build_graph_for_chrom, vcf_path, chrom)

        # 4) Collect results
        for chrom, fut in futures.items():
            vg_bytes = fut.result()
            out_file = f"{output_prefix}_{chrom}.vg"
            with open(out_file, "wb") as f:
                f.write(vg_bytes)
            print(f"Wrote {out_file}")

    print(f"Successfully built .vg files for {len(contigs)} chromosome(s).")


# If you want to run as a script:
if __name__ == "__main__":
    vcf_input = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample.vcf.gz"
    output_prefix = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto"
    vcf_to_vg_haplotypes(vcf_input, output_prefix)
