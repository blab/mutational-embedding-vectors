# The (current) purpose of this branch is for exploratory analysis

# Cross-Pathogen PLM Embedding Analysis

This project uses protein language models (PLMs) to analyze evolutionary dynamics across multiple viral pathogens. The pipeline computes mutation-level embeddings, evolutionary velocity, and comparative metrics to understand patterns of viral evolution across pathogens with varying rates of adaptation.

Adaptive substitution rates are from [Kistler & Bedford 2023](https://github.com/blab/adaptive-evolution): "An atlas of continuous adaptive evolution in endemic human viruses." Cell Host Microbe 31: 1-12.

# Installation

Install dependencies with pip including PyTorch and transformers:

```
pip install -r requirements.txt
```

# Workflow

The workflow is managed through Snakemake and processes multiple pathogens in parallel:

## Run Full Pipeline

Process all configured pathogens through the complete analysis:
```
snakemake --cores all all_pathogens
```

## Individual Steps

Run specific analysis steps:

```
# Download Nextstrain data only
snakemake --cores 1 data/pathogen/{pathogen}/auspice.json

# Compute embeddings only
snakemake --cores 4 results/pathogen/{pathogen}/node_embeddings.pkl

# Generate final summary
snakemake --cores 1 aggregate_pathogens
```

## Pathogen-specific Analysis

To analyze specific pathogens:
```
# Single pathogen
snakemake --cores 4 results/cross_pathogen/sars_cov_2_spike/mutation_metrics.tsv

# Multiple pathogens
snakemake --cores 8 results/cross_pathogen/{influenza_h3n2_ha,rsv_a_g}/velocity.tsv
```

## Available Datasets

The following pathogens are pre-configured in `config/cross_pathogen_config.yaml`:

- `influenza_h3n2_ha`: Influenza A/H3N2 HA1 protein (8.62 adaptive subs/year)
- `influenza_h1n1pdm_ha`: Influenza A/H1N1pdm HA1 protein (5.38 adaptive subs/year)
- `norovirus_gii4_vp1`: Norovirus GII.4 VP1 capsid protein (4.12 adaptive subs/year)
- `cov_229e_s1`: 229E coronavirus S1 protein (2.94 adaptive subs/year)
- `rsv_a_g`: RSV-A G protein (2.77 adaptive subs/year)
- `rsv_b_g`: RSV-B G protein (2.35 adaptive subs/year)
- `influenza_vic_ha`: Influenza B/Victoria HA1 protein (2.16 adaptive subs/year)
- `oc43_a_s1`: OC43 coronavirus S1 protein (2.12 adaptive subs/year)
- `sars_cov_2_spike`: SARS-CoV-2 Spike protein (?? adaptive subs/year)
- `mumps_sh`: Mumps SH protein (-0.17 adaptive subs/year)
- `measles_n`: Measles N protein (-0.62 adaptive subs/year, internal protein)

# Configuration

Edit `config/pathogen_config.yaml` to configure pathogens and settings:

```yaml
pathogens:
  sars_cov_2_spike:
    dataset: "https://nextstrain.org/ncov/gisaid/global/all-time"
    gene: "S"
    protein: "Spike"
    protein_coords: "S:1-1273"
    surface: true
    adaptive_subs_per_year: 0.4 # To Be Upadated
  
pipeline_settings:
  embedding_model: "facebook/esm2_t6_8M_UR50D"
  min_recurrence: 3
  batch_size: 100
```

# Output Structure

```
results/cross_pathogen/
├── {pathogen}/
│   ├── node_embeddings.pkl       # PLM embeddings
│   ├── mutation_vectors.tsv      # Embedding space changes per mutation
│   ├── mutation_metrics.tsv      # Mutation-level statistics
│   └── velocity.tsv              # Evolutionary velocity per branch
└── summary/
    └── pathogen_summary.csv      # Cross-pathogen comparison
```

# License

This repository is licensed under the MIT License. See the LICENSE file for details.
