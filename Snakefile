# Get cross-pathogen analyses from config
configfile: "config/pathogen_config.yaml"
PATHOGENS = list(config.get("pathogens", {}).keys())

# Main rule to run the entire cross-pathogen pipeline
rule all_pathogens:
    input:
        expand("results/pathogen/{pathogen}/mutation_metrics.tsv", pathogen=PATHOGENS),
        expand("results/pathogen/{pathogen}/velocity.tsv", pathogen=PATHOGENS),
        "results/pathogen/summary/pathogen_summary.csv"

# Download Nextstrain data
rule download_pathogen_tree:
    output:
        tree = "data/pathogen/{pathogen}/auspice.json"
    params:
        dataset = lambda w: config["pathogens"][w.pathogen]["dataset"]
    shell:
        """
        nextstrain remote download {params.dataset:q} {output.tree:q}
        """

# Extract sequences (alignment)
rule prepare_pathogen_alignment:
    input:
        auspice = "data/pathogen/{pathogen}/auspice.json"
    output:
        alignment = "data/pathogen/{pathogen}/alignment.fasta"
    params:
        gene = lambda w: config["pathogens"][w.pathogen]["gene"]
    shell:
        """
        python scripts/alignment.py \
            --json {input.auspice:q} \
            --output {output.alignment:q} \
            --gene {params.gene:q}
        """

# Extract branches
rule prepare_pathogen_branches:
    input:
        auspice = "data/pathogen/{pathogen}/auspice.json",
        alignment = "data/pathogen/{pathogen}/alignment.fasta"
    output:
        branches = "data/pathogen/{pathogen}/branches.tsv"
    shell:
        """
        python scripts/branches.py \
            --json {input.auspice:q} \
            --alignment {input.alignment:q} \
            --output {output.branches:q}
        """

# Compute node embeddings
rule compute_plm_embeddings:
    input:
        alignment = "data/pathogen/{pathogen}/alignment.fasta"
    output:
        embeddings = "results/pathogen/{pathogen}/node_embeddings.pkl"
    params:
        protein_coords = lambda w: config["pathogens"][w.pathogen]["protein_coords"],
        model = lambda w: config.get("plm_settings", {}).get("model", "facebook/esm2_t6_8M_UR50D")
    shell:
        """
        python scripts/compute_node_embeddings.py \
            --sequences {input.alignment:q} \
            --output {output.embeddings:q} \
            --protein-coords {params.protein_coords:q} \
            --model {params.model:q}
        """

# Analyze mutations and compute vectors
rule analyze_plm_mutations:
    input:
        auspice = "data/pathogen/{pathogen}/auspice.json",
        alignment = "data/pathogen/{pathogen}/alignment.fasta",
        embeddings = "results/pathogen/{pathogen}/node_embeddings.pkl",
        branches = "data/pathogen/{pathogen}/branches.tsv"
    output:
        vectors = "results/pathogen/{pathogen}/mutation_vectors.tsv"
    params:
        protein_coords = lambda w: config["pathogens"][w.pathogen]["protein_coords"]
    shell:
        """
        python scripts/analyze_mutation_esm.py \
            --tree {input.auspice:q} \
            --sequences {input.alignment:q} \
            --embeddings {input.embeddings:q} \
            --output {output.vectors:q} \
            --protein-coords {params.protein_coords:q}
        """

# Step 6: Compute mutation metrics (R_m, A_m, L_m)
rule compute_mutation_metrics:
    input:
        vectors = "results/pathogen/{pathogen}/mutation_vectors.tsv"
    output:
        metrics = "results/pathogen/{pathogen}/mutation_metrics.tsv"
    params:
        min_recurrence = lambda w: config.get("plm_settings", {}).get("min_recurrence", 3)
    shell:
        """
        python scripts/analyze_vectors.py \
            --vectors {input.vectors:q} \
            --output {output.metrics:q} \
            --min-recurrence {params.min_recurrence}
        """

# Step 7: Compute evolutionary velocity
rule compute_evolutionary_velocity:
    input:
        auspice = "data/pathogen/{pathogen}/auspice.json",
        embeddings = "results/pathogen/{pathogen}/node_embeddings.pkl"
    output:
        velocity = "results/pathogen/{pathogen}/velocity.tsv"
    shell:
        """
        python scripts/mutation_analysis/compute_velocity.py \
            --tree {input.auspice:q} \
            --embeddings {input.embeddings:q} \
            --output {output.velocity:q}
        """

# Step 8: Aggregate results across all pathogens
rule aggregate_pathogens:
    input:
        metrics = expand("results/pathogen/{pathogen}/mutation_metrics.tsv", pathogen=PATHOGENS),
        velocity = expand("results/pathogen/{pathogen}/velocity.tsv", pathogen=PATHOGENS)
    output:
        summary = "results/pathogen/summary/pathogen_summary.csv"
    params:
        config_file = "config/pathogen_config.yaml",
        results_dir = "results/pathogen",
        output_dir = "results/pathogen/summary"
    shell:
        """
        python scripts/analyze_pathogens.py \
            --config {params.config_file:q} \
            --results {params.results_dir:q} \
            --output {params.output_dir:q}
        """
