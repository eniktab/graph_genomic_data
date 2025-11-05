import os
import pysam
from concurrent.futures import ProcessPoolExecutor
import src.old.proto.graph_pb2 as graph_pb2

# Mapping from DNA bases to enum values
BASE_MAP = {
    'A': graph_pb2.BASE_A,
    'C': graph_pb2.BASE_C,
    'G': graph_pb2.BASE_G,
    'T': graph_pb2.BASE_T
}


class PhasedNucleotideGraphBuilder:
    """Builds a graph for a single chromosome using Python data structures.

    A 'graph' is built incrementally from VCF records. Once complete, the data
    can be serialized into a protobuf (Graph) message.

    Attributes:
        attributes (dict[str, str]): Key-value pairs describing graph metadata.
        nodes (dict[str, dict]): A map of node_id -> node_data.
        edges (dict[tuple[str, str], dict]): A map of (source_id, target_id) ->
            edge_data, capturing traversals.
        last_node (dict[str, dict]): Tracks the last visited node per
            (sample, phase_set, haplotype_index).
    """

    __slots__ = ("attributes", "nodes", "edges", "last_node")

    def __init__(self, chromosome_name: str = None) -> None:
        """Initializes the graph builder.

        Args:
            chromosome_name (str): Optional chromosome name for the graph.
                If provided, it is stored under self.attributes["chromosome"].
        """
        self.attributes = {}
        if chromosome_name:
            self.attributes["chromosome"] = chromosome_name

        self.nodes = {}
        self.edges = {}
        self.last_node = {}

        self._create_start_node()

    def _create_start_node(self) -> None:
        """Creates a special 'START' node in the graph.

        This node serves as an initial anchor for edges that have no prior node.
        """
        self.nodes["START"] = {
            "id": "START",
            "base": graph_pb2.BASE_UNSPECIFIED,
            "attributes": {}
        }

    def _add_node_if_absent(self,
                            node_id: str,
                            base: int = graph_pb2.BASE_UNSPECIFIED,
                            attributes: dict = None) -> str:
        """Adds a node if it doesn't already exist.

        Merges additional attributes if the node is already present.

        Args:
            node_id (str): Unique identifier for the node.
            base (int): The node's base, mapped to `graph_pb2.BASE_*`.
            attributes (dict): Optional dictionary of attributes to attach
                or merge into the node.

        Returns:
            str: The node_id (unmodified).
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "base": base,
                "attributes": dict(attributes) if attributes else {}
            }
        else:
            if attributes:
                self.nodes[node_id]["attributes"].update(attributes)
        return node_id

    def _add_edge_if_absent(self, source_id: str, target_id: str) -> dict:
        """Adds an edge if it doesn't already exist, otherwise returns existing.

        Args:
            source_id (str): Source node identifier.
            target_id (str): Target node identifier.

        Returns:
            dict: A dictionary representing edge data for this edge.
        """
        key = (source_id, target_id)
        if key not in self.edges:
            self.edges[key] = {
                "source": source_id,
                "target": target_id,
                "observation_count": 0,
                "individuals": set()
            }
        return self.edges[key]

    def process_variant_record(self, rec_data: dict, samples: list) -> dict:
        """Processes a single VCF record (in dict form) and returns sample paths.

        The input `rec_data` is a Python dictionary containing fields extracted
        from a pysam VCF record, such as:
            {
                'pos': int,
                'ref': str,
                'alts': tuple[str, ...],
                'samples': {
                    sample_name: {'GT': tuple[int], 'PS': str or None},
                    ...
                }
            }

        If no samples carry an ALT allele, returns None.

        Args:
            rec_data (dict): A Python dict of record fields (pos, ref, alts, etc.).
            samples (list[str]): A list of sample names from the VCF.

        Returns:
            dict or None: A structure describing paths for each sample, keyed by
            sample name, then by phase set, e.g.:

                sample_paths[sample][phase_set] = [
                    [node_ids_for_haplotype_0],
                    [node_ids_for_haplotype_1]
                ]

            Returns None if no alt allele is found for any sample.
        """
        pos = rec_data['pos']
        ref = rec_data['ref']
        alts = rec_data['alts']
        if not alts:
            return None

        # Determine if any sample has an ALT
        has_alt = False
        for sample in samples:
            call = rec_data['samples'].get(sample, {})
            gt = call.get('GT')
            if gt and any(a is not None and a > 0 for a in gt):
                has_alt = True
                break
        if not has_alt:
            return None

        # Create nodes for REF and ALTs
        ref_id = f"{pos}:{ref}"
        self._add_node_if_absent(
            ref_id,
            base=BASE_MAP.get(ref, graph_pb2.BASE_UNSPECIFIED),
            attributes={"is_alt": "0"}
        )

        alt_ids = []
        for alt in alts:
            alt_id = f"{pos}:{alt}"
            self._add_node_if_absent(
                alt_id,
                base=BASE_MAP.get(alt, graph_pb2.BASE_UNSPECIFIED),
                attributes={"is_alt": "1"}
            )
            alt_ids.append(alt_id)

        # Gather chosen nodes
        sample_paths = {}
        for sample in samples:
            call = rec_data['samples'].get(sample, {})
            gt = call.get('GT') or ()
            if not gt:
                continue
            ps = call.get('PS') or "D"

            if sample not in sample_paths:
                sample_paths[sample] = {}
            if ps not in sample_paths[sample]:
                sample_paths[sample][ps] = [[], []]

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

        return sample_paths if sample_paths else None

    def add_paths_to_graph(self, sample_paths: dict) -> None:
        """Updates the internal graph using the provided sample paths.

        For each sample and phase set, edges are formed from the last node to
        the current node (or 'START' if no last node exists). The observation
        count is incremented for each edge, and the sample is added to the
        edge's individuals set.

        Args:
            sample_paths (dict): Nested structure describing each sample's
                paths for the current variant, keyed by sample and phase set.
        """
        for sample, ps_dict in sample_paths.items():
            if sample not in self.last_node:
                self.last_node[sample] = {}

            for ps, haplotypes in ps_dict.items():
                if ps not in self.last_node[sample]:
                    self.last_node[sample][ps] = [None, None]

                for haplotype_idx, node_ids in enumerate(haplotypes):
                    for node_id in node_ids:
                        prev_node = self.last_node[sample][ps][haplotype_idx]
                        if prev_node is None:
                            edge_info = self._add_edge_if_absent("START", node_id)
                        else:
                            edge_info = self._add_edge_if_absent(prev_node, node_id)

                        edge_info["observation_count"] += 1
                        edge_info["individuals"].add(sample)

                        self.last_node[sample][ps][haplotype_idx] = node_id

    def build_proto(self) -> graph_pb2.Graph:
        """Converts the builder's internal data into a Graph protobuf message.

        Returns:
            graph_pb2.Graph: A protobuf Graph object representing the graph
            built so far.
        """
        graph = graph_pb2.Graph()

        # Graph attributes
        for k, v in self.attributes.items():
            graph.attributes[k] = v

        # Create node protos
        for node_id, node_data in self.nodes.items():
            node_proto = graph.nodes.add()
            node_proto.id = node_data["id"]
            node_proto.base = node_data["base"]
            for ak, av in node_data["attributes"].items():
                node_proto.attributes[ak] = av

        # Create edge protos
        for (source_id, target_id), edge_data in self.edges.items():
            edge_proto = graph.edges.add()
            edge_proto.source = edge_data["source"]
            edge_proto.target = edge_data["target"]
            edge_proto.observation_count = edge_data["observation_count"]

            for individual in sorted(edge_data["individuals"]):
                edge_proto.individuals.append(individual)

        return graph


def _build_graph_for_chromosome(chrom: str, records: list, samples: list) -> bytes:
    """Builds and returns a serialized protobuf for one chromosome.

    Args:
        chrom (str): The chromosome name.
        records (list[dict]): List of variant record dictionaries. Each dict
            contains Python-native fields for pos, ref, alts, etc.
        samples (list[str]): List of sample names from the VCF file.

    Returns:
        bytes: The serialized bytes of the resulting Graph protobuf.
    """
    builder = PhasedNucleotideGraphBuilder(chromosome_name=chrom)
    for rec_data in records:
        spaths = builder.process_variant_record(rec_data, samples)
        if spaths:
            builder.add_paths_to_graph(spaths)
    return builder.build_proto().SerializeToString()


def vcf_to_proto_by_chromosome(
    vcf_input: str,
    output_prefix: str,
    use_parallel: bool = True,
    max_workers: int = None
) -> None:
    """Converts a phased VCF to protobuf graphs, grouped by chromosome.

    Reads a VCF file using pysam, converts each record into a plain
    Python dictionary, groups them by chromosome, and then builds a
    separate protobuf graph per chromosome. The resulting protobufs
    are written to disk.

    Args:
        vcf_input (str): Path to the input VCF file.
        output_prefix (str): Output prefix for the resulting .pb files.
        use_parallel (bool): Whether to use multiprocessing to build graphs
            in parallel. Defaults to True.
        max_workers (int): Maximum number of parallel workers. Defaults to
            `os.cpu_count()` if not specified.
    """
    with pysam.VariantFile(vcf_input) as vcf_in:
        samples = list(vcf_in.header.samples)

        records_by_chrom = {}
        record_count = 0
        for rec in vcf_in:
            record_count += 1
            chrom = rec.chrom
            if chrom not in records_by_chrom:
                records_by_chrom[chrom] = []

            # Convert the pysam.VariantRecord into a dict that is pickleable.
            rec_data = {
                "pos": rec.pos,
                "ref": rec.ref,
                "alts": rec.alts,  # tuple of alt alleles
                "samples": {}
            }
            # Extract relevant sample-level data
            for sample in samples:
                call = rec.samples[sample]
                rec_data["samples"][sample] = {
                    "GT": call.get("GT"),
                    "PS": call.get("PS")
                }

            records_by_chrom[chrom].append(rec_data)

    if use_parallel:
        max_workers = max_workers or os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for chrom, recs in records_by_chrom.items():
                future_map[chrom] = executor.submit(
                    _build_graph_for_chromosome, chrom, recs, samples
                )
            # Collect results
            for chrom, fut in future_map.items():
                graph_bytes = fut.result()
                out_file = f"{output_prefix}_{chrom}.pb"
                with open(out_file, "wb") as f:
                    f.write(graph_bytes)
                print(f"Wrote {out_file}")
    else:
        # Serial approach
        for chrom, recs in records_by_chrom.items():
            graph_bytes = _build_graph_for_chromosome(chrom, recs, samples)
            out_file = f"{output_prefix}_{chrom}.pb"
            with open(out_file, "wb") as f:
                f.write(graph_bytes)
            print(f"Wrote {out_file}")

    print(f"Processed {record_count} records in {vcf_input}")

"""
if __name__ == "__main__":
    vcf_input = "/path/to/Phased_sample.vcf"
    output_prefix = "/path/to/phased_sample_proto"
    vcf_to_proto_by_chromosome(vcf_input, output_prefix, use_parallel=True)
"""



# Example usage
if __name__ == "__main__":
    vcf_input = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/Phased_sample.vcf"
    output_prefix = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto"
    vcf_to_proto_by_chromosome(vcf_input, output_prefix, use_parallel=True)
