import src.old.proto.graph_pb2 as graph_pb2


def create_path_table(proto_file: str, output_file: str = "path_table_output.txt") -> None:
    """
    Parses a Graph protobuf file, enumerates all possible paths, and writes them to an output file.

    The output format follows:
        START -> A -> A (NA00001 & NA00002)
        START -> C -> A (NA00003)
        START -> A -> G (NA00003)

    Paths with empty intersections of samples are still included with empty parentheses ().

    Args:
        proto_file (str): Path to the binary protobuf file containing the graph data.
        output_file (str): Path to the output text file where paths will be stored.
    """
    # Parse the Protobuf Graph
    graph_msg = graph_pb2.Graph()
    with open(proto_file, "rb") as f:
        graph_msg.ParseFromString(f.read())

    # Build adjacency list
    adjacency = {}
    for edge in graph_msg.edges:
        edge_data = {
            "individuals": set(edge.individuals),
            "observation_count": edge.observation_count,
        }
        adjacency.setdefault(edge.source, []).append((edge.target, edge_data))

    print(f"Total edges in the graph: {len(graph_msg.edges)}")
    print(f"Adjacency keys found: {list(adjacency.keys())}")

    def dfs_all_paths(start: str):
        """Finds all paths from START to leaf nodes using DFS."""
        stack = [(start, [start])]
        all_paths = []
        while stack:
            node, path = stack.pop()
            if node not in adjacency:
                all_paths.append(path)
            else:
                for next_node, _ in adjacency[node]:
                    if next_node not in path:  # Avoid cycles
                        stack.append((next_node, path + [next_node]))
        return all_paths

    all_paths = dfs_all_paths("START")
    print(f"Number of paths found from 'START': {len(all_paths)}")

    def short_label(node_id: str) -> str:
        """Extracts the last part of a node ID for display."""
        return "START" if node_id == "START" else node_id.split(":")[-1]

    with open(output_file, "w") as out_file:
        written_count = 0
        for path in all_paths:
            if len(path) < 2:
                print(f"Skipping path {path} due to no edges.")
                continue

            samples_in_all_edges = None
            for u, v in zip(path, path[1:]):
                edge_data_list = adjacency.get(u, [])
                edge_data = next((data for target, data in edge_data_list if target == v), None)
                samples_in_edge = edge_data["individuals"] if edge_data else set()
                samples_in_all_edges = samples_in_edge if samples_in_all_edges is None else samples_in_all_edges & samples_in_edge

            path_str = " -> ".join(short_label(n) for n in path)
            sample_str = " & ".join(sorted(samples_in_all_edges)) if samples_in_all_edges else ""
            line_str = f"{path_str} ({sample_str})"
            out_file.write(line_str + "\n")
            written_count += 1

        print(f"Wrote {written_count} path lines to {output_file}")


if __name__ == "__main__":
    proto_file = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto_19.vg"
    output_file = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto_19.txt"
    create_path_table(proto_file, output_file)
