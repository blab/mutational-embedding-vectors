import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_mutation_vectors(file_path):
    """Loads mutation vectors from TSV."""
    try:
        df = pd.read_csv(file_path, sep="\t")
        if df.empty:
            return df
        # Parse diff_vector string "v1,v2,..." to numpy array
        df["vector"] = df["diff_vector"].apply(lambda x: np.fromstring(x, sep=","))
        return df
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return pd.DataFrame()


def compute_mutation_stats(df, min_recurrence=3, eps=1e-12, center_for_rank=True):
    """
    Compute stats for all mutations in the dataframe.
    Returns a dataframe with one row per mutation type (e.g., N501Y).

    Assumes df has:
      - "vector": array-like of shape (d,) per row
    Optionally:
      - "mutation" or ("wt_aa","pos","mut_aa")
      - "llr"
    """

    results = []

    # Construct mutation name if missing
    if "mutation" not in df.columns:
        if "pos" in df.columns and "wt_aa" in df.columns and "mut_aa" in df.columns:
            df = df.copy()
            df["mutation"] = df["wt_aa"] + df["pos"].astype(str) + df["mut_aa"]
        else:
            print(
                "Error: Could not identify mutation names (need 'mutation' or wt_aa/pos/mut_aa)."
            )
            return pd.DataFrame()

    grouped = df.groupby("mutation")
    print(f"Analyzing {len(grouped)} unique mutations...")

    for mutation, group in grouped:
        n_m = len(group)
        if n_m < min_recurrence:
            continue

        # vectors: (n, d)
        vectors = np.stack(group["vector"].values).astype(np.float64, copy=False)

        #  Magnitude stats
        norms = np.linalg.norm(vectors, axis=1)
        Am = float(np.mean(norms))
        mag_median = float(np.median(norms))
        mag_cv = float(np.std(norms, ddof=1) / (Am + eps)) if n_m > 1 else np.nan

        # Unit directions (n, d)
        U = vectors / (norms[:, None] + eps)

        #  Direction consistency
        # Pairwise cosine similarity (signed) and axial (abs)
        C = U @ U.T

        if n_m > 1:
            iu = np.triu_indices(n_m, k=1)
            pair_cos = C[iu]  # all off-diagonal cosines

            dir_pair_mean = float(np.mean(pair_cos))  # [-1, 1]
            dir_pair_median = float(np.median(pair_cos))  # [-1, 1]
            ax_pair_mean = float(np.mean(np.abs(pair_cos)))  # [0, 1]
            ax_pair_median = float(np.median(np.abs(pair_cos)))  # [0, 1]
        else:
            dir_pair_mean = dir_pair_median = np.nan
            ax_pair_mean = ax_pair_median = np.nan

        #  Subspace / low-rank structure
        # Use SVD energy on U so it’s not dominated by magnitude outliers.
        X = U

        if n_m > 1:
            s = np.linalg.svd(X, compute_uv=False)
            energy = s**2
            total = float(np.sum(energy) + eps)
            rank1_energy = float(
                energy[0] / total
            )  # [0,1], fraction captured by top singular direction

            # Effective rank: exp(entropy of normalized singular value energy)
            p = energy / total
            eff_rank = float(np.exp(-np.sum(p * np.log(p + eps))))
        else:
            rank1_energy = np.nan
            eff_rank = np.nan

        # LLR (if available)
        mean_llr = float(group["llr"].mean()) if "llr" in group.columns else np.nan

        results.append(
            {
                "mutation": mutation,
                "n": int(n_m),
                # Magnitude
                "Am": Am,
                "mag_median": mag_median,
                "mag_cv": mag_cv,
                # Direction
                "dir_pair_mean": dir_pair_mean,
                "dir_pair_median": dir_pair_median,
                "ax_pair_mean": ax_pair_mean,
                "ax_pair_median": ax_pair_median,
                # Subspace structure (replacement for PCA EVR[0])
                "rank1_energy": rank1_energy,
                "eff_rank": eff_rank,
                # Auxiliary
                "llr": mean_llr,
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mutation metrics from vectors"
    )
    parser.add_argument("--vectors", required=True, help="Path to mutation_vectors.tsv")
    parser.add_argument("--output", required=True, help="Path to output metrics.tsv")
    parser.add_argument(
        "--min-recurrence", type=int, default=3, help="Minimum recurrence to analyze"
    )

    args = parser.parse_args()

    print(f"Loading vectors from {args.vectors}...")
    df = load_mutation_vectors(args.vectors)

    if df.empty:
        print("No data found.")
        return

    print("Computing metrics...")
    metrics_df = compute_mutation_stats(df, min_recurrence=args.min_recurrence)

    print(f"Computed metrics for {len(metrics_df)} mutations.")
    metrics_df.to_csv(args.output, sep="\t", index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
