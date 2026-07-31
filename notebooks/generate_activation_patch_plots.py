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

pio.get_chrome()

TOK_DIR = "./covfit_stuff/Tokenizer"
CONF_DIR = "./covfit_stuff/Config"
TASK_IDS_FILE = "./covfit_stuff/task_id_dict.pt"
FOLD_ID = 0
N_TARGETS = 1565
MODEL_PATH = f"./covfit_stuff/models/covfit_model_20241007_{FOLD_ID}.ckpt"

# MODEL_PATH = "TheSatoLab-UTokyo/CoVFit"
# FOLD_IDS_TO_USE = [0]
# TARGET_FOLD_ID = 0
# OUTPUT_PREFIX = "inference_results"

model_name = "facebook/esm2_t33_650M_UR50D"
device = experiment_config.device
CONTEXT_LEN = 1024
torch.autograd.grad_mode.set_grad_enabled(False)
torch.set_float32_matmul_precision("medium")

esm_fine_tuned = interp_utils.get_model()

esm_fine_tuned = esm_fine_tuned.to(device)
esm_fine_tuned = esm_fine_tuned.eval()

esm_config = esm_fine_tuned.config
esm_config.token_dropout = False
esm_config.model_name = model_name
REPO_ID = esm_config.model_name
original_task_id_infos = torch.load("./covfit_stuff/task_id_dict.pt", map_location=device)

tokenizer_config = {}
special_tokens_map_file = "./covfit_stuff/Tokenizer/special_tokens_map.json"
tokenizer_config["vocab_file"] = "./covfit_stuff/Tokenizer/vocab.txt"
tokenizer_config["model_max_length"] = CONTEXT_LEN

with open("./covfit_stuff/Tokenizer/special_tokens_map.json", "r") as f:
    tokenizer_config = {**tokenizer_config, **(json.load(f))}

with open(tokenizer_config["vocab_file"], "r") as f:
    f_data = f.read().split("\n")
    aa_to_toks_map = {i:f_data[i] for i in range(len(f_data))}
    aa_to_toks_map_rev = {aa_to_toks_map[k]:k for k in aa_to_toks_map.keys()}

tokenizer = EsmTokenizer(**tokenizer_config)

hooked_esm_config = interp_utils.get_hooked_esm_config(esm_config, context_len=CONTEXT_LEN, use_hook_tokens=True)
hooked_esm = HookedTransformer(hooked_esm_config)
print(hooked_esm.load_state_dict(interp_utils.get_hooked_state_dict(esm_fine_tuned.state_dict(), hooked_esm_config)))

# clean up memory
torch.cuda.empty_cache()

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
d_out_vocab = esm_fine_tuned.regressor[3].weight.size(0)
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

hooked_esm.reset_hooks(including_permanent=True)
hooked_esm = interp_utils.add_perma_hooks_to_mask_pad_tokens(hooked_esm, 1)

component_name_map = dict()
for l in range(esm_config.num_hidden_layers + 1):
    if l < esm_config.num_hidden_layers:
        component_name_map[l] = f"blocks.{l}.hook_resid_pre"
    
    # final layer
    elif l == esm_config.num_hidden_layers:
        component_name_map[l] = f"unembed.hook_in"

def get_logit_hooked(output: Float[Tensor, "batch pos d_model"], tok_id):
    logits = interp_utils.get_logits_hooked_esm(output[:,0,:], esm_fine_tuned.regressor)[:,tok_id]
    torch.cuda.empty_cache()
    return logits

def get_rev_names(id_seq):
    """
    Given seq x in all_uniq_seqs, get the corresponding name(s) of sequences that have the same spike protein
    """
    if type(id_seq) == int:
        id_seq = [id_seq]

    rev_name_dict = {}
    for id_s in id_seq:
        name_idx = np.argwhere(np.array(seq_idxs) == id_seq)[:,0]
        rev_name_dict[id_s] = [seq_names[x] for x in name_idx]
    return rev_name_dict

logit_id = original_task_id_infos["fitness_USA"]

relevant_mutations = [
    ("G339H", lambda x: x <= "21M"),
    ("R346T", lambda x: "XBB" in x),
    ("K417N", lambda x: True), 
    ("V445P", lambda x: x <= "23I"),
    ("L455F", lambda x: True),
    ("F456L", lambda x: x.endswith("(XBB.1.5)")),
    ("E484A", lambda x: x <= "21M"),
    ("S486P", lambda x: "XBB" in x),
    ("Q493R", lambda x: x <= "21M"),
    ("P681H", lambda x: x <= "23I"),
]

wt_mut_seq_pairs = []
np.random.seed(0)
for mut, seq_selector in relevant_mutations:
    wt_resid = mut[0]
    mut_resid = mut[-1]
    site = int(mut[1:-1])

    uniq_seqs = np.unique([all_uniq_seqs[seq_idxs[i]] for i,n in enumerate(seq_names) if seq_selector(name_to_clade_dict[n])])
    rand_seqs = np.random.permutation(uniq_seqs)

    seq_orig_idx = [x for x in rand_seqs if x[site-1] == wt_resid][:80]
    seq_new_idx = [x[:site-1] + mut_resid + x[site:] for x in seq_orig_idx]

    print(f"Mutation = {wt_resid} {site} {mut_resid}; {seq_orig_idx[10][site-1]}, {seq_new_idx[10][site-1]}; {len(seq_orig_idx)} total seqs")

    seq_orig_toks = tokenizer(seq_orig_idx, return_tensors="pt", return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=MAX_LEN).input_ids.to(device)
    seq_new_toks = tokenizer(seq_new_idx, return_tensors="pt", return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=MAX_LEN).input_ids.to(device)

    wt_mut_seq_pairs.append((mut, seq_orig_toks, seq_new_toks))

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

for mut, seq_orig_toks, seq_new_toks in wt_mut_seq_pairs:
    print(mut)

    hooked_esm.reset_hooks(including_permanent=False)
    corr_toks = seq_orig_toks
    clean_toks = seq_new_toks
    corrupted_logit_mean = get_logit_hooked(hooked_esm(corr_toks), logit_id).mean().item()
    clean_logit_mean = get_logit_hooked(hooked_esm(clean_toks), logit_id).mean().item()
    print(corrupted_logit_mean - clean_logit_mean)
    torch.cuda.empty_cache()
    
    patched_head_output_comps = []
    for receiver_input in ["k", "q", "v", "z", "pattern"]:
        if receiver_input == "pattern":
            _, corrupted_cache = hooked_esm.run_with_cache(corr_toks, names_filter = lambda x: ("hook_q" in x) or ("hook_k" in x))
            del _
            torch.cuda.empty_cache()
        else:
            _, corrupted_cache = hooked_esm.run_with_cache(corr_toks, names_filter = lambda x: f"hook_{receiver_input}" in x)
            del _
            torch.cuda.empty_cache()
    
        patched_head_output = interp_utils.get_act_patch_attn_head_out_all_pos(
            hooked_esm, 
            logit_id=logit_id, 
            clean_tokens=clean_toks, 
            corrupted_cache=corrupted_cache,
            receiver_input=receiver_input,
            patching_metric=functools.partial(path_patching_metric, corrupted_logit_mean=corrupted_logit_mean, clean_logit_mean=clean_logit_mean),
            get_logit_hooked=get_logit_hooked
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
