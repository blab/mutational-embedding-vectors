import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

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


def load_branches(branches_file):
    """Load branches from TSV file."""
    return pd.read_csv(branches_file, sep="\t")


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
            return dna_sequence.replace("-", "").replace("*", "")

        _, coords = protein_coords.split(":")
        start, end = map(int, coords.split("-"))

        # Convert to 0-indexed python slicing
        # Start is 1-indexed, so start-1
        # End is inclusive, so end (python slice is exclusive)
        dna_segment = dna_sequence[start - 1 : end]

        # Remove gaps for translation
        dna_clean = dna_segment.replace("-", "")

        from Bio.Seq import Seq

        protein_seq = str(Seq(dna_clean).translate())

        return protein_seq.replace("*", "")  # Remove stop codon if present

    except Exception as e:
        print(f"Error extracting protein: {e}")
        return None


def compute_llr(sequence, pos, wt_aa, mut_aa, tokenizer, model):
    """
    Compute Log-Likelihood Ratio (LLR) for a mutation.
    LLR = log P(mut) - log P(wt)
    """
    try:
        # Tokenize
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get mask token index
        mask_token_id = tokenizer.mask_token_id

        # Create masked input
        # ESM tokenizer adds CLS token at start, so pos becomes pos+1
        token_pos = pos + 1

        if token_pos >= inputs["input_ids"].shape[1] - 1:  # Check bounds (minus EOS)
            return None

        original_token_id = inputs["input_ids"][0, token_pos].item()
        inputs["input_ids"][0, token_pos] = mask_token_id

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Get probabilities at the masked position
        probs = torch.softmax(logits[0, token_pos], dim=0)

        # Get token IDs for WT and Mut
        wt_token = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_token = tokenizer.convert_tokens_to_ids(mut_aa)

        log_p_wt = torch.log(probs[wt_token]).item()
        log_p_mut = torch.log(probs[mut_token]).item()

        return log_p_mut - log_p_wt

    except Exception as e:
        # print(f"LLR Error: {e}")
        return None


def get_esm_embedding(protein_seq, tokenizer, model):
    """Compute mean embedding for protein sequence."""
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
            outputs = model(**inputs, output_hidden_states=True)

        # Use last hidden state
        # outputs.hidden_states[-1] is the same as outputs.last_hidden_state
        embeddings = outputs.hidden_states[-1][0, 1:-1, :]  # Remove CLS/EOS
        mean_embedding = torch.mean(embeddings, dim=0)

        return mean_embedding.cpu().numpy()

    except Exception as e:
        return None


def analyze_mutation_esm(
    fasta_file,
    branches_file,
    output_file,
    protein_coords,
    model_name="facebook/esm2_t6_8M_UR50D",
    node_embeddings=None,
    skip_llr=False,
):
    print(f"Loading ESM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use ForMaskedLM to get logits for LLR
    model = EsmForMaskedLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    if skip_llr:
        print(
            "Skipping LLR calculation (would need proper sequence LL implementation)..."
        )

    print("Loading sequences...")
    sequences = load_sequences(fasta_file)

    print("Loading branches...")
    branches = load_branches(branches_file)

    results = []

    print(f"Scanning branches for mutations in {protein_coords}...")

    for _, row in tqdm(branches.iterrows(), total=len(branches)):
        parent_id = row["parent"]
        child_id = row["child"]

        if parent_id not in sequences or child_id not in sequences:
            continue

        parent_dna = sequences[parent_id]
        child_dna = sequences[child_id]

        # Extract protein sequences
        parent_prot = get_protein_sequence(parent_dna, protein_coords)
        child_prot = get_protein_sequence(child_dna, protein_coords)

        if parent_prot is None or child_prot is None:
            print(f"Failed to extract protein for {parent_id} or {child_id}")
            continue

        if len(parent_prot) != len(child_prot):
            # Indel - skip for now
            # print(f"Length mismatch: {len(parent_prot)} vs {len(child_prot)}")
            continue

        # Find mutations
        diffs = []
        for i, (p, c) in enumerate(zip(parent_prot, child_prot)):
            if p != c:
                diffs.append((i, p, c))

        # Only analyze single mutants for cleanliness, or handle multiples?
        # Let's handle all, but compute LLR for each single change

        if len(diffs) > 0:
            # Get embeddings (pre-computed or compute on fly)
            emb_parent = None
            emb_child = None

            if (
                node_embeddings is not None
                and parent_id in node_embeddings
                and child_id in node_embeddings
            ):
                emb_parent = node_embeddings[parent_id]
                emb_child = node_embeddings[child_id]
            else:
                # Fallback to compute on fly
                emb_parent = get_esm_embedding(parent_prot, tokenizer, model)
                emb_child = get_esm_embedding(child_prot, tokenizer, model)

            if emb_parent is None or emb_child is None:
                continue

            diff_vector = emb_child - emb_parent

            for pos, wt, mut in diffs:
                # Compute LLR (CURRENTLY INCORRECT - skipping for now)
                # TODO: Implement proper sequence-level LL comparison
                llr = None
                if not skip_llr:
                    llr = compute_llr(parent_prot, pos, wt, mut, tokenizer, model)

                results.append(
                    {
                        "parent_id": parent_id,
                        "child_id": child_id,
                        "pos": pos + 1,  # 1-indexed
                        "wt_aa": wt,
                        "mut_aa": mut,
                        "mutation": f"{wt}{pos + 1}{mut}",
                        "llr": llr,
                        "diff_vector": ",".join(map(str, diff_vector)),
                    }
                )

    print(f"Found {len(results)} mutation events.")

    df = pd.DataFrame(results)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze ESM embedding differences and LLR for mutations"
    )
    parser.add_argument(
        "--tree",
        required=True,
        help="Path to Auspice JSON tree (not used directly here, but for compatibility)",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to node embeddings PKL (not used, computed on fly)",
    )
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument(
        "--protein-coords", required=True, help="Protein coordinates (Name:Start-End)"
    )
    parser.add_argument("--sequences", help="Path to FASTA sequences")
    parser.add_argument(
        "--skip-llr",
        action="store_true",
        help="Skip LLR calculation (currently incorrect implementation)",
    )

    args = parser.parse_args()

    # Load pre-computed embeddings if available
    import pickle

    node_embeddings = {}
    if os.path.exists(args.embeddings):
        print(f"Loading pre-computed embeddings from {args.embeddings}...")
        with open(args.embeddings, "rb") as f:
            node_embeddings = pickle.load(f)
    else:
        print(f"Warning: Embeddings file {args.embeddings} not found. Will recompute.")

    analyze_mutation_esm(
        args.sequences,
        args.tree.replace("auspice.json", "branches.tsv"),
        args.output,
        args.protein_coords,
        node_embeddings=node_embeddings,
        skip_llr=args.skip_llr,
    )
