import argparse
import os
import pickle
import sys

import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, EsmConfig, EsmForMaskedLM
from huggingface_hub import hf_hub_download

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_sequences(fasta_file):
    """Load sequences from FASTA file into a dictionary."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_str = str(record.seq).upper()
        sequences[record.id] = seq_str
        # Also store cleaned ID if needed
        clean_id = record.id.split("|")[0].strip()
        sequences[clean_id] = seq_str
    return sequences


def get_protein_sequence(dna_sequence, protein_coords):
    """
    Extract and translate the specific protein region from the DNA sequence.
    protein_coords format: "Name:Start-End" (1-indexed, inclusive)

    If the sequence is already amino acids, return it directly.
    If it's DNA, extract the region and translate.
    """
    try:
        # Check if sequence is already amino acids (contains non-ACGT characters)
        seq_upper = dna_sequence.upper().replace("-", "").replace("N", "")
        is_dna = all(c in "ACGT" for c in seq_upper[:100] if c.isalpha())

        if not is_dna:
            # Already amino acid sequence - return as is
            # return dna_sequence.replace("-", "").replace("*", "") for now, don't remove gaps
            return dna_sequence.replace("*", "")

        # DNA sequence - extract region and translate
        _, coords = protein_coords.split(":")
        start, end = map(int, coords.split("-"))

        # Convert to 0-indexed python slicing
        dna_segment = dna_sequence[start - 1 : end]

        # Remove gaps for translation
        # dna_clean = dna_segment.replace("-", "") for now, don't remove gaps

        from Bio.Seq import Seq

        protein_seq = str(Seq(dna_clean).translate())

        return protein_seq.replace("*", "")  # Remove stop codon

    except Exception as e:
        print(f"Error extracting protein: {e}")
        return None


def compute_embeddings(
    sequences, output_file, protein_coords, model_name="facebook/esm2_t33_650M_UR50D"
):
    print(f"Loading ESM model: {model_name}")
    esm_config = EsmConfig.from_pretrained(model_config.da_model_name)
    esm_config.token_dropout = False
    esm_config.model_name = model_name

    REPO_ID = esm_config.model_name
    special_tokens_map_file = "special_tokens_map.json"
    tokenizer_config = {}
    tokenizer_config["vocab_file"] = hf_hub_download(repo_id=REPO_ID, filename="vocab.txt")
    tokenizer_config["model_max_length"] = CONTEXT_LEN
    with open(hf_hub_download(repo_id=REPO_ID, filename=special_tokens_map_file), "r") as f:
        tokenizer_config = {**tokenizer_config, **(json.load(f))}

    tokenizer = EsmTokenizer(**tokenizer_config)
    esm_cov_state_dict_path = "../notebooks/plm_circuits/covfit_stuff/model_ESM2_coronaviridae/pytorch_model.bin"
    model = EsmForMaskedLM(esm_config).to(device)
    esm_disp_state_dict = torch.load(esm_cov_state_dict_path)
    del esm_disp_state_dict["esm.embeddings.position_embeddings.weight"]
    del esm_disp_state_dict["esm.embeddings.position_ids"]
    model.load_state_dict(esm_disp_state_dict)
    model.eval()

    embeddings = {}

    print(f"Computing embeddings for {len(sequences)} sequences...")

    count = 0
    for name, seq in tqdm(sequences.items()):
        # Extract protein
        protein_seq = get_protein_sequence(seq, protein_coords)

        if protein_seq is None or len(protein_seq) < 3:
            continue

        try:
            inputs = tokenizer(
                protein_seq,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Mean pooling
            emb = outputs.last_hidden_state[0, 1:-1, :]
            mean_emb = torch.mean(emb, dim=0).cpu().numpy()

            embeddings[name] = mean_emb
            count += 1

        except Exception as e:
            # print(f"Error embedding {name}: {e}")
            continue

    print(f"Computed {count} embeddings.")

    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute node embeddings for a specific protein"
    )
    parser.add_argument(
        "--sequences", required=True, help="Path to FASTA sequences (DNA)"
    )
    parser.add_argument("--output", required=True, help="Output PKL file")
    parser.add_argument(
        "--protein-coords", required=True, help="Protein coordinates (Name:Start-End)"
    )
    parser.add_argument(
        "--model", default="facebook/esm2_t6_8M_UR50D", help="ESM model name"
    )

    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    compute_embeddings(sequences, args.output, args.protein_coords, args.model)
