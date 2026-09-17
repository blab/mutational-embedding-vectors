import argparse
import os
import pickle
import sys
import typing

import pandas as pd
import torch
from Bio import SeqIO
from typing import List, Union, Optional, Callable, Sequence
from transformers import (
    EsmForMaskedLM,
    EsmConfig,
    PretrainedConfig,
    EsmTokenizer,
    DataCollatorForLanguageModeling,
    Trainer
)

from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sklearn.linear_model import LinearRegression

import einops
import yaml
import sys
import json
import functools
import os
import shutil

import numpy as np
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import math
from tqdm import tqdm

from matplotlib import pyplot as plt

from jaxtyping import Bool, Float, Int
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
from plotly_utils import (
    imshow,
    line,
    bar
)

import circuitsvis as cv
from IPython.display import display, HTML
from IPython import get_ipython

sys.path.append("../config")
import experiment_config

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)

# Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

sys.path.append("../scripts")
from compute_node_embeddings import load_sequences, get_protein_sequence
from branches import json_to_tree
import interp_utils

from covfit_stuff.config import Config, ModelConfig
from covfit_stuff.esm_regression import load_model_for_inference, get_model_predictions, EsmForRegression
import tempfile

torch.autograd.grad_mode.set_grad_enabled(False)
torch.set_float32_matmul_precision("medium")
# small thing to turn off annoying wand questions
os.environ["WANDB_DISABLED"] = "true"
CONTEXT_LEN = experiment_config.CONTEXT_LEN
device = experiment_config.device
print(f"device = {device}")

esm_config, _esm_covfit, (hooked_esm_covfit, get_logit_covfit) = interp_utils.setup_and_load_model(experiment_config, "covfit")
del _esm_covfit
torch.cuda.empty_cache()

tokenizer_config = {}
tokenizer_config["vocab_file"] = experiment_config.VOCAB_FILE
tokenizer_config["model_max_length"] = experiment_config.CONTEXT_LEN

with open(experiment_config.SPECIAL_TOK_MAP, "r") as f:
    tokenizer_config = {**tokenizer_config, **(json.load(f))}

with open(tokenizer_config["vocab_file"], "r") as f:
    f_data = f.read().split("\n")
    aa_to_toks_map = {i:f_data[i] for i in range(len(f_data))}
    aa_to_toks_map_rev = {aa_to_toks_map[k]:k for k in aa_to_toks_map.keys()}

tokenizer = EsmTokenizer(**tokenizer_config)
original_task_id_infos = torch.load(experiment_config.TASK_IDS_FILE, map_location=device)

def tokenizer_for_map(seq, seq_key="input_ids", tokenizer=tokenizer): #Tokenizer and params including special_tokens_mask required for MLM
    return tokenizer(
        seq[seq_key],
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        padding="max_length",
        max_length=300,
    )

# data loading
with open("../config/pathogen_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
pathogens = list(config["pathogens"].keys())
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,return_tensors='pt',mlm_probability=0.15)

MAX_LEN=1024
pathogen_suffixes = ["africa", "asia", "europe", "north_america", "oceania", "south_america"]
pathogen_name = "sars_cov_2_spike"
protein_coords = config["pathogens"][f"{pathogen_name}_africa"]["protein_coords"]

name_to_clade_dict = dict()
for suff in pathogen_suffixes:
    print(suff)
    with open(f"../data/pathogen/{pathogen_name}_{suff}/auspice.json", "r") as f:
        tree_json = json.load(f)
        test_tree = json_to_tree(tree_json)
        nodes = list(test_tree.find_clades(order="postorder"))
        name_to_clade_dict.update({n.name:n.node_attrs["clade_membership"]["value"] for n in nodes})

"""
all_uniq_seqs - seqs used in training
seq_names - names of ALL sequences
all_seqs - ALL sequences
seq_idxs - map from seq_names to uniq_seqs, i.e. seq_names[i] is for uniq_seqs[seq_idxs[i]]
"""

all_seqs = []
seq_names = []
seq_idxs = []
all_uniq_seqs = []

for suff in pathogen_suffixes:
    fasta_file = f"../data/pathogen/{pathogen_name}_{suff}/alignment.fasta"
    data = load_sequences(fasta_file)
    sequence_names, sequences = list(zip(*list(data.items())))
    sequences = [get_protein_sequence(x, protein_coords) for x in sequences]

    keep_idx = [i for i,x in enumerate(sequences) if len(x.replace("-","")) > (CONTEXT_LEN // 5) * 4]
    sequences = [sequences[i] for i in keep_idx]
    sequence_names = [sequence_names[i] for i in keep_idx]

    uniq_seqs_suff, unique_inv_idx  = np.unique(sequences, return_inverse=True) # For the purpose of eval, I only care about unique sequences

    all_seqs.extend(sequences)
    seq_names.extend(sequence_names)
    seq_idxs.extend(unique_inv_idx + len(all_uniq_seqs))
    all_uniq_seqs.extend(uniq_seqs_suff)

all_uniq_seqs, unique_inv_idx  = np.unique(all_uniq_seqs, return_inverse=True) # For the purpose of eval, I only care about unique sequences
seq_idxs = [unique_inv_idx[idx] for idx in seq_idxs]
all_uniq_seqs = list(all_uniq_seqs)

# identical code to how it's compute_node_embeddings.py
tok_output = tokenizer(all_uniq_seqs, return_tensors="pt", return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=MAX_LEN)
tok_seqs = tok_output.input_ids.to(device)
tok_masks = tok_output.attention_mask.to(device)

print(pathogen_name)
print(f"Number unique sequences: {len(all_uniq_seqs)}")
print(tok_seqs.shape)

relevant_mutations = [
    ("G339H", lambda x: x <= "21M"),
    ("R346T", lambda x: "XBB" in x),
    ("K417N", lambda x: True),
    ("V445P", lambda x: x <= "23I"),
    ("L455F", lambda x: True),
    ("F456L", lambda x: x.endswith("(XBB.1.5)")),
    ("E484A", lambda x: x <= "21M"),
    ("S486P", lambda x: "XBB" in x or "JN" in x or "21" in x),
    ("Q493R", lambda x: x <= "21M"),
    ("P681H", lambda x: x <= "23I"),
]

wt_mut_seq_pairs = dict() # key=mutation, contains the clipped-to-80 pairs of (wt, mut, <mask>) seqs
wt_seq_all_pairs = dict() # key=mutation, contains all pairs of (wt, mut, <mask>) seqsnp.random.seed(0)

np.random.seed(0)
for mut, seq_selector in relevant_mutations:
    wt_resid = mut[0]
    mut_resid = mut[-1]
    site = int(mut[1:-1])

    uniq_seqs = np.unique([all_uniq_seqs[seq_idxs[i]] for i,n in enumerate(seq_names) if seq_selector(name_to_clade_dict[n])])
    rand_seqs = np.random.permutation(uniq_seqs)

    all_seq_orig_idx = [x for x in rand_seqs if x[site-1] == wt_resid]
    all_seq_new_idx = [x[:site-1] + mut_resid + x[site:] for x in all_seq_orig_idx]
    all_seq_mask_idx = [x[:site-1] + "<mask>" + x[site:] for x in all_seq_orig_idx]

    print(f"Mutation = {wt_resid} {site} {mut_resid}; {all_seq_orig_idx[10][site-1]}, {all_seq_new_idx[10][site-1]}; {len(all_seq_orig_idx)} total seqs (clipped to 80)")

    all_seq_orig_toks = tokenizer(all_seq_orig_idx, return_tensors="pt", return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=MAX_LEN).input_ids.to(device)
    all_seq_new_toks = tokenizer(all_seq_new_idx, return_tensors="pt", return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=MAX_LEN).input_ids.to(device)
    all_seq_mask_toks = tokenizer(all_seq_mask_idx, return_tensors="pt", return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=MAX_LEN).input_ids.to(device)

    wt_mut_seq_pairs[mut] = (all_seq_orig_toks[:80], all_seq_new_toks[:80], all_seq_mask_toks[:80])
    wt_seq_all_pairs[mut] = (all_seq_orig_toks, all_seq_new_toks, all_seq_mask_toks)

    sanity_check = (wt_mut_seq_pairs[mut][0] !=  wt_mut_seq_pairs[mut][1]).sum(dim=0)
    sanity_check_mask = (wt_mut_seq_pairs[mut][0] !=  wt_mut_seq_pairs[mut][2]).sum(dim=0)
    print(torch.arange(all_seq_orig_toks.shape[-1])[sanity_check.to(bool).cpu()].item(), sanity_check[site].item(), "\n")
    print(torch.arange(all_seq_orig_toks.shape[-1])[sanity_check_mask.to(bool).cpu()].item(), sanity_check_mask[site].item(), "\n")

# path patching metric 
def path_patching_metric(
    logits: Float[Tensor, "batch"],
    corrupted_logit_mean: float,
    clean_logit_mean: float,
):
    """
    Equals 0 when performance is conserved (i.e. high fitness sequences still high fitness)
    Equals -1 when performance is destroyed (high fitness sequence is degraded)
    """

    return ((logits - clean_logit_mean) / (clean_logit_mean - corrupted_logit_mean)).mean().item()

# activation patching for attn. heads
print("beginning activation patching (attn. heads!)")
for mut, _ in relevant_mutations:
    seq_orig_toks, seq_new_toks, _ = wt_mut_seq_pairs[mut]
    hooked_esm_covfit.reset_hooks(including_permanent=False)

    corr_toks = seq_orig_toks
    clean_toks = seq_new_toks

    corrupted_logit_mean = get_logit_covfit(hooked_esm_covfit(corr_toks), logit_id).mean().item()
    clean_logit_mean = get_logit_covfit(hooked_esm_covfit(clean_toks), logit_id).mean().item()
    print(corrupted_logit_mean - clean_logit_mean)
    torch.cuda.empty_cache()

    patched_head_output_comps = []
    for receiver_input in ["k", "q", "v", "z", "pattern"]:
        if receiver_input == "pattern":
            _, corrupted_cache = hooked_esm_covfit.run_with_cache(corr_toks, names_filter = lambda x: ("hook_q" in x) or ("hook_k" in x))
            del _
            torch.cuda.empty_cache()
        else:
            _, corrupted_cache = hooked_esm_covfit.run_with_cache(corr_toks, names_filter = lambda x: f"hook_{receiver_input}" in x)
            del _
            torch.cuda.empty_cache()
    
        patched_head_output = interp_utils.get_act_patch_attn_head_out_all_pos(
            hooked_esm_covfit, 
            logit_id=logit_id, 
            clean_tokens=clean_toks, 
            corrupted_cache=corrupted_cache,
            receiver_input=receiver_input,
            patching_metric=functools.partial(path_patching_metric, corrupted_logit_mean=corrupted_logit_mean, clean_logit_mean=clean_logit_mean),
            get_logit_hooked=get_logit_covfit
        )
        
        patched_head_output_comps.append(patched_head_output)
        del corrupted_cache
        torch.cuda.empty_cache()

    patched_head_output_tensor = torch.stack(patched_head_output_comps, dim=0)

    fig = imshow(
        patched_head_output_tensor,
        labels={"x": "Head", "y": "Layer", "color": "Change in fitness"},
        title="Activation patching change in fitness (low fitness into high fitness)",
        width=1400,
        height=600,
        facet_labels=["Key", "Query", "Value", "Z", "pattern"],
        facet_col=0,
        return_fig=True
        # range_color=(-0.8,0.8)
    )
    fig.write_image(f"../figures/{mut}_activation_patch.png", width=1400, height=600, scale=2)

# 
print("beginning activation patching (MLP)")
for mut, _ in relevant_mutations:
    seq_orig_toks, seq_new_toks, _ = wt_mut_seq_pairs[mut]
    hooked_esm_covfit.reset_hooks(including_permanent=False)

    corr_toks = seq_orig_toks
    clean_toks = seq_new_toks

    corrupted_logit_mean = get_logit_covfit(hooked_esm_covfit(corr_toks), logit_id).mean().item()
    clean_logit_mean = get_logit_covfit(hooked_esm_covfit(clean_toks), logit_id).mean().item()
    print(corrupted_logit_mean - clean_logit_mean)
    torch.cuda.empty_cache()

    patched_head_output_comps = []
    for receiver_input in ["k", "q", "v", "z", "pattern"]:
        if receiver_input == "pattern":
            _, corrupted_cache = hooked_esm_covfit.run_with_cache(corr_toks, names_filter = lambda x: ("hook_q" in x) or ("hook_k" in x))
            del _
            torch.cuda.empty_cache()
        else:
            _, corrupted_cache = hooked_esm_covfit.run_with_cache(corr_toks, names_filter = lambda x: f"hook_{receiver_input}" in x)
            del _
            torch.cuda.empty_cache()
    
        patched_head_output = interp_utils.get_act_patch_attn_head_out_all_pos(
            hooked_esm_covfit, 
            logit_id=logit_id, 
            clean_tokens=clean_toks, 
            corrupted_cache=corrupted_cache,
            receiver_input=receiver_input,
            patching_metric=functools.partial(path_patching_metric, corrupted_logit_mean=corrupted_logit_mean, clean_logit_mean=clean_logit_mean),
            get_logit_hooked=get_logit_covfit
        )
        
        patched_head_output_comps.append(patched_head_output)
        del corrupted_cache
        torch.cuda.empty_cache()

    patched_head_output_tensor = torch.stack(patched_head_output_comps, dim=0)

    fig = imshow(
        patched_head_output_tensor,
        labels={"x": "Head", "y": "Layer", "color": "Change in fitness"},
        title="Activation patching change in fitness (low fitness into high fitness)",
        width=1400,
        height=600,
        facet_labels=["Key", "Query", "Value", "Z", "pattern"],
        facet_col=0,
        return_fig=True
        # range_color=(-0.8,0.8)
    )
    fig.write_image(f"../figures/{mut}_activation_patch.png", width=1400, height=600, scale=2)
