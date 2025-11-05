import src.old.proto.graph_pb2 as graph_pb2

def parse_pb_file(pb_filename: str) -> None:
    """
    Reads and prints the contents of a serialized Graph protobuf file.

    Args:
        pb_filename (str): Path to the protobuf file to be parsed.
    """
    # Create an empty Graph message
    graph = graph_pb2.Graph()

    # Read the .pb file in binary mode
    try:
        with open(pb_filename, "rb") as f:
            graph.ParseFromString(f.read())
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading file {pb_filename}: {e}")
        return

    # Print global attributes
    print("Global Graph Attributes:")
    for key, value in graph.attributes.items():
        print(f"  {key} = {value}")

    # Print nodes
    print("\nNodes:")
    for i, node in enumerate(graph.nodes, start=1):
        print(f"Node #{i}:")
        print(f"  id: {node.id}")
        print(f"  base: {node.base}")  # This is an enum value
        if node.attributes:
            print("  attributes:")
            for k, v in node.attributes.items():
                print(f"    {k} = {v}")

    # Print edges
    print("\nEdges:")
    for i, edge in enumerate(graph.edges, start=1):
        print(f"Edge #{i}:")
        print(f"  source: {edge.source}")
        print(f"  target: {edge.target}")
        print(f"  observation_count: {edge.observation_count}")
        if edge.individuals:
            print("  individuals: " + ", ".join(edge.individuals))
        if edge.attributes:
            print("  attributes:")
            for k, v in edge.attributes.items():
                print(f"    {k} = {v}")

if __name__ == "__main__":
    pb_filename = "/home/niktabel/workspace/media/gadi_g_te53/en9803/data/sandpit/phased_sample_proto_19.pb"
    parse_pb_file(pb_filename)
