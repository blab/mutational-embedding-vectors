import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_pathogen_data(pathogen_name, pathogen_config, results_dir):
    """Load metrics and velocity for a pathogen."""
    pathogen_dir = Path(results_dir) / pathogen_name

    metrics_path = pathogen_dir / "mutation_metrics.tsv"
    velocity_path = pathogen_dir / "velocity.tsv"

    data = {
        "name": pathogen_name,
        "gene": pathogen_config["gene"],
        "adaptive_subs_per_year": pathogen_config["adaptive_subs_per_year"],
        "surface": pathogen_config["surface"],
    }

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, sep="\t")
        data["n_mutations"] = len(metrics_df)
    else:
        print(f"Warning: No metrics found for {pathogen_name}")
        return None

    if velocity_path.exists():
        velocity_df = pd.read_csv(velocity_path, sep="\t")
        data["mean_velocity"] = velocity_df["velocity"].mean()
        data["median_velocity"] = velocity_df["velocity"].median()
    else:
        print(f"Warning: No velocity found for {pathogen_name}")
        data["mean_velocity"] = np.nan

    return data


def analyze_cross_pathogen(config_path, results_dir, output_dir):
    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_data = []

    for name, pathogen_config in config["pathogens"].items():
        data = load_pathogen_data(name, pathogen_config, results_dir)
        if data:
            summary_data.append(data)

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "pathogen_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate cross-pathogen results and run tests"
    )
    parser.add_argument(
        "--config",
        default="config/cross_pathogen_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--results", default="results/cross_pathogen", help="Results directory"
    )
    parser.add_argument(
        "--output",
        default="results/cross_pathogen/summary",
        help="Output directory for summary",
    )

    args = parser.parse_args()

    analyze_cross_pathogen(args.config, args.results, args.output)
