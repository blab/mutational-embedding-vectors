import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from Bio import Phylo


def load_embeddings(embeddings_path):
    """Load node embeddings from pickle file."""
    import pickle

    with open(embeddings_path, "rb") as f:
        return pickle.load(f)


def load_tree_json(tree_path):
    """Load Auspice JSON tree."""
    with open(tree_path, "r") as f:
        return json.load(f)


def get_date(node_data):
    """Extract date from node data."""
    # Auspice JSON format varies
    # Usually in node_data['node_attrs']['num_date']['value']
    try:
        return node_data["node_attrs"]["num_date"]["value"]
    except KeyError:
        return None


def compute_velocity(tree_json, embeddings, output_file):
    """
    Compute evolutionary velocity for each branch.
    Velocity = Euclidean Distance(Parent, Child) / Time Delta
    """

    # Traverse tree to find parent-child pairs
    # Auspice JSON is nested

    results = []

    def traverse(node, parent_name=None, parent_date=None):
        node_name = node["name"]
        node_date = get_date(node)

        # print(f"Visiting {node_name}, Date: {node_date}, Parent: {parent_name}")

        if parent_name and parent_name in embeddings and node_name in embeddings:
            # Calculate distance
            emb_parent = embeddings[parent_name]
            emb_child = embeddings[node_name]

            distance = np.linalg.norm(emb_child - emb_parent)

            # Calculate time delta
            if parent_date is not None and node_date is not None:
                time_delta = node_date - parent_date

                print(
                    f"  Branch {parent_name}->{node_name}: Dist={distance:.4f}, Time={time_delta:.4f}"
                )

                # Avoid division by zero or very small time deltas
                if time_delta > 0.001:
                    velocity = distance / time_delta

                    results.append(
                        {
                            "parent": parent_name,
                            "child": node_name,
                            "distance": distance,
                            "time_delta": time_delta,
                            "velocity": velocity,
                        }
                    )
            else:
                print(f"  Missing dates for {parent_name}->{node_name}")
        elif parent_name:
            if parent_name not in embeddings:
                print(f"  Parent {parent_name} not in embeddings")
            if node_name not in embeddings:
                print(f"  Child {node_name} not in embeddings")

        # Recurse
        if "children" in node:
            for child in node["children"]:
                traverse(child, node_name, node_date)

    traverse(tree_json["tree"])

    print(f"Computed velocity for {len(results)} branches.")

    df = pd.DataFrame(results)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved velocity results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute evolutionary velocity from embeddings and tree"
    )
    parser.add_argument("--tree", required=True, help="Path to Auspice JSON tree")
    parser.add_argument(
        "--embeddings", required=True, help="Path to node embeddings PKL"
    )
    parser.add_argument("--output", required=True, help="Output TSV file")

    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings)
    tree = load_tree_json(args.tree)

    compute_velocity(tree, embeddings, args.output)
