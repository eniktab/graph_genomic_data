import src.old.proto.graph_pb2 as graph_pb2

def create_path_table(proto_file, output_file="path_table_output.txt"):
    """
    Reads the given protobuf Graph file, enumerates all possible paths,
    and writes lines like:
        START -> A -> A (NA00001 & NA00002)
        START -> C -> A (NA00003)
        START -> A -> G (NA00003)
    to the specified output file.

    This debug-friendly version:
      - Prints diagnostic information about the graph structure,
        number of paths found, etc.
      - Does NOT skip paths if they have an empty intersection of samples.
        Instead, we write them out with empty parentheses '()'.
    """

    # ------------------------------------------------
    # 1) Parse the Protobuf Graph
    # ------------------------------------------------
    graph_msg = graph_pb2.Graph()
    with open(proto_file, "rb") as f:
        graph_msg.ParseFromString(f.read())

    # ------------------------------------------------
    # 2) Build an adjacency list for quick traversal
    #    adjacency[source_id] = [(target_id, edge_data), ...]
    # ------------------------------------------------
    adjacency = {}
    for edge in graph_msg.edges:
        source = edge.source
        target = edge.target
        # We'll store the set of samples + any other info
        edge_data = {
            "individuals": set(edge.individuals),
            "observation_count": edge.observation_count
        }
        if source not in adjacency:
            adjacency[source] = []
        adjacency[source].append((target, edge_data))

    # Debug: Print how many edges and adjacency keys
    print(f"Total edges in the graph: {len(graph_msg.edges)}")
    print("Adjacency keys found:", list(adjacency.keys()))

    # ------------------------------------------------
    # 3) Find all paths from "START" to "leaves" (nodes with no outgoing edges)
    #    We'll do a simple DFS that enumerates all possible paths.
    # ------------------------------------------------
    def dfs_all_paths(start):
        stack = [(start, [start])]  # each item: (current_node, path_so_far)
        all_paths = []
        while stack:
            node, path = stack.pop()
            # If node has no outgoing edges, it's a leaf => record this path
            if node not in adjacency or len(adjacency[node]) == 0:
                all_paths.append(path)
            else:
                # Explore each edge
                for (next_node, _) in adjacency[node]:
                    # Avoid cycles
                    if next_node not in path:
                        stack.append((next_node, path + [next_node]))
        return all_paths

    all_paths = dfs_all_paths("START")
    print(f"Number of paths found from 'START': {len(all_paths)}")

    # ------------------------------------------------
    # 4) For each path, compute the intersection of 'individuals'
    #    across the edges in that path.
    # ------------------------------------------------
    def short_label(node_id):
        """Transforms a node ID like '19:111:A' into just 'A' for printing."""
        if node_id == "START":
            return "START"
        # Example node_id: "19:111:A" => return just the last part ("A")
        parts = node_id.split(":")
        return parts[-1]

    # ------------------------------------------------
    # 5) Write results to the output file
    #    We do NOT skip paths with empty intersections.
    # ------------------------------------------------
    with open(output_file, "w") as out_file:
        written_count = 0  # Track how many lines are actually written
        for path in all_paths:
            # If there's no edges in the path (e.g. "START" alone?), skip
            if len(path) < 2:
                print(f"Skipping path {path} because it has no edges.")
                continue

            # Gather intersection of individuals across the edges
            samples_in_all_edges = None

            # We'll iterate over pairs of consecutive nodes in the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data_list = adjacency[u]
                edge_data = None
                for (target, data) in edge_data_list:
                    if target == v:
                        edge_data = data
                        break
                if edge_data is None:
                    # If something is malformed in the graph
                    samples_in_edge = set()
                else:
                    samples_in_edge = edge_data["individuals"]

                if samples_in_all_edges is None:
                    samples_in_all_edges = samples_in_edge
                else:
                    samples_in_all_edges = samples_in_all_edges.intersection(samples_in_edge)

            # Prepare the path string
            path_str = " -> ".join(short_label(n) for n in path)

            # Even if samples_in_all_edges is empty or None, we still write the path
            if not samples_in_all_edges:
                # Debug: let user know it's empty, but we won't skip it
                print(f"Path {path} has empty intersection of samples;")
                sample_str = ""
            else:
                # Format the samples with & if multiple
                sample_str = " & ".join(sorted(samples_in_all_edges))

            line_str = f"{path_str} ({sample_str})"
            out_file.write(line_str + "\n")
            written_count += 1

        print(f"Wrote {written_count} path lines to {output_file}")
# Example usage:
if __name__ == "__main__":
    proto_file = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto_19.pb"
    create_path_table(proto_file, "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto_19.txt")

