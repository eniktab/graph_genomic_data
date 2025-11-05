import os
from neo4j import GraphDatabase
import src.old.proto.graph_pb2 as graph_pb2


#################################
# 1. Neo4j Connection + Setup   #
#################################

# Adjust the credentials/URI as needed
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"



driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_constraints():
    """
    Create a uniqueness constraint so that
    we do not ingest duplicate nodes by 'id'.
    """
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:Node)
            REQUIRE n.id IS UNIQUE
        """)

#########################################
# 2. Functions to Ingest Proto into Neo4j
#########################################

def ingest_graph_proto(proto_file_path: str):
    """
    Parse a Graph protobuf from disk and ingest into Neo4j.
    """
    # Parse the protobuf
    graph_message = graph_pb2.Graph()
    with open(proto_file_path, "rb") as f:
        graph_message.ParseFromString(f.read())

    # Convert proto to Python objects
    # We'll store them in two arrays for node ingestion and edge ingestion.
    nodes_to_ingest = []
    edges_to_ingest = []

    # 2.1 Prepare the node data
    for node in graph_message.nodes:
        # node.id (string), node.base (enum), node.attributes (map<string, string>)
        # We'll store the base as a string, e.g. "A", "C", "G", "T", "UNSPECIFIED"
        base_str = _base_enum_to_str(node.base)

        # Flatten attributes into a dictionary of { key: value }, ensuring they’re all strings
        # (graph_pb2 gives attributes as a dict of <str, str> so we can use them as-is.)
        node_attribs = dict(node.attributes)

        # Build the dictionary for MERGE
        node_data = {
            "id": node.id,
            "base": base_str,
            **node_attribs  # merges the node attributes into top-level props
        }
        nodes_to_ingest.append(node_data)

    # 2.2 Prepare the edge data
    for edge in graph_message.edges:
        # edge.source, edge.target, edge.observation_count, edge.individuals (list of strings)
        edge_data = {
            "source": edge.source,
            "target": edge.target,
            "observation_count": edge.observation_count,
            "individuals": list(edge.individuals),  # repeated string -> Python list
        }
        edges_to_ingest.append(edge_data)

    # 2.3 Ingest nodes and edges
    with driver.session() as session:
        # (A) Ingest all nodes
        _create_nodes(session, nodes_to_ingest)

        # (B) Ingest all edges
        _create_edges(session, edges_to_ingest)


def _create_nodes(session, nodes):
    """
    Use MERGE to create (or match if already exists) nodes by their unique 'id'.
    Then set/merge all properties on them.
    """
    query = """
    UNWIND $batch AS row
    MERGE (n:Node {id: row.id})
    SET n += row
    """
    # We pass the entire "row" dictionary so that `SET n += row` merges
    # all properties (base, plus any 'attributes') onto the node.
    session.run(query, parameters={"batch": nodes})

def _create_edges(session, edges):
    """
    Ingest edges using MERGE on source/target nodes (already created),
    then MERGE the relationship, and set properties.
    """
    query = """
    UNWIND $batch AS row
    MATCH (src:Node {id: row.source}), (tgt:Node {id: row.target})
    MERGE (src)-[r:EDGE]->(tgt)
    SET r.observation_count = coalesce(r.observation_count, 0) + row.observation_count,
        r.individuals = coalesce(r.individuals, []) + row.individuals
    """
    # Explanation:
    #   - We MATCH nodes by their id. They must already exist or it fails.
    #   - MERGE the relationship (src)-[r:EDGE]->(tgt).
    #   - For "observation_count", we add the new count to the existing
    #     so that if the edge is repeated multiple times, we accumulate it.
    #   - For "individuals", we combine the existing list with the new list
    #     (this will create duplicates if the same individual is in multiple sets,
    #      you may want to call APOC's `apoc.coll.toSet(...)` or handle duplicates differently).
    session.run(query, parameters={"batch": edges})

def _base_enum_to_str(base_enum_value):
    """
    Helper to map the `graph_pb2.BASE_*` enum to a string like "A", "C", "G", "T", ...
    """
    mapping = {
        1: "A",  # graph_pb2.BASE_A
        2: "C",  # graph_pb2.BASE_C
        3: "G",  # graph_pb2.BASE_G
        4: "T",  # graph_pb2.BASE_T
        0: "UNSPECIFIED"  # BASE_UNSPECIFIED (or handle how you like)
    }
    return mapping.get(base_enum_value, "UNK")


##################################
# 3. Example of Path Search Query #
##################################

def find_individuals_with_path(path_string: str):
    """
    Given a path string like "ACC", find sequences of Node(base='A') -> Node(base='C') -> Node(base='C')
    in the graph and return the distinct individuals that appear on those edges.
    """
    # For a path length of 3 bases (e.g., "ACC"), we want 2 edges:
    query = """
    MATCH (n1:Node)-[r1:EDGE]->(n2:Node)-[r2:EDGE]->(n3:Node)
    WHERE n1.base = $b1
      AND n2.base = $b2
      AND n3.base = $b3
    WITH (r1.individuals + r2.individuals) AS combinedIndivs
    UNWIND combinedIndivs AS indiv
    WITH DISTINCT indiv AS distinctIndiv
    RETURN collect(distinctIndiv) AS individuals
    """
    b1, b2, b3 = list(path_string)  # e.g. "ACC" -> ['A','C','C']

    with driver.session() as session:
        results = session.run(query, parameters={"b1": b1, "b2": b2, "b3": b3})
        records = results.single()
        if records:
            return records["individuals"]  # This is a list of distinct individuals
        else:
            return []

############################################
# 4. Putting It All Together (Example Main) #
############################################

if __name__ == "__main__":
    # 1. Ensure constraints exist
    create_constraints()

    # 2. Ingest a Protobuf file (as produced by your vcf_to_proto_by_chromosome)
    #    Suppose you have a file: "output_chr1.pb"
    proto_file = "output_chr1.pb"
    if os.path.exists(proto_file):
        ingest_graph_proto(proto_file)

    # 3. Example usage: find individuals for path "ACC"
    #    (Adjust the path length or search logic to match your needs.)
    individuals = find_individuals_with_path("ACC")
    print("Individuals on path 'ACC':", individuals)




