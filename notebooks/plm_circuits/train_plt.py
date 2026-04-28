import argparse
import os
import pickle
import sys
import typing

import pandas as pd

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

from covfit_stuff.config import Config, ModelConfig
from covfit_stuff.esm_regression import load_model_for_inference, get_model_predictions, EsmForRegression
import tempfile

sys.path.append("../../scripts")
from compute_node_embeddings import load_sequences, get_protein_sequence
from interp_utils import get_hooked_state_dict, get_hooked_esm_config, get_logits_hooked_esm, get_fairesm_state_dict

sys.path.append("../../ProtoMech/")
sys.path.append("../../ProtoMech/training_transcoder")
sys.path.append("../../ProtoMech/training")
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import SequenceDataModule
from plt_module import PLTLightningModule
import random

#########################################################
#               LOADING Model                           #
#########################################################
TOK_DIR = "./covfit_stuff/Tokenizer"
CONF_DIR = "./covfit_stuff/Config"
TASK_IDS_FILE = "./covfit_stuff/task_id_dict.pt"
FOLD_ID = 0
N_TARGETS = 1565
MODEL_PATH = f"./covfit_stuff/models/covfit_model_20241007_{FOLD_ID}.ckpt"

model_name = "facebook/esm2_t33_650M_UR50D"
device = "cuda"
CONTEXT_LEN = 1024
torch.autograd.grad_mode.set_grad_enabled(False)
torch.set_float32_matmul_precision("medium")

def get_model(
    TOK_DIR = "./covfit_stuff/Tokenizer",
    CONF_DIR = "./covfit_stuff/Config",
    TASK_IDS_FILE = "./covfit_stuff/task_id_dict.pt",
    FOLD_ID = 0,
    N_TARGETS = 1565,
    MODEL_PATH = f"./covfit_stuff/models/covfit_model_20241007_{FOLD_ID}.ckpt",
    device=device
):
    esm_config = EsmConfig.from_pretrained(CONF_DIR)
    model = EsmForRegression(esm_config, N_TARGETS).to(device)

    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        target_modules=["key", "query", "value","dense"],
        lora_dropout=0.05,
        bias="lora_only",
        modules_to_save=["regressor"]
    )
    esm_fine_tuned = get_peft_model(model, lora_config)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # keys_to_remove = []
    # for key in state_dict.keys():
    #     if 'contact_head' in key:
    #         keys_to_remove.append(key)
    
    # for key in keys_to_remove:
    #     del state_dict[key]

    wrong_keys = [key for key in state_dict.keys() if key not in esm_fine_tuned.state_dict().keys()]
    key_list = list(state_dict.keys())
    for key in key_list:
        if key in wrong_keys:
            correct_key = key.rsplit('.',1)[0]+'.base_layer.'+key.rsplit('.',1)[1]
            state_dict[correct_key] = state_dict.pop(key)

    del state_dict["base_model.model.esm.embeddings.position_embeddings.base_layer.weight"]
    
    esm_fine_tuned.load_state_dict(state_dict)
    esm_fine_tuned = esm_fine_tuned.merge_and_unload()
    esm_fine_tuned.eval()
    esm_fine_tuned.esm.embeddings.token_dropout = False

    return esm_fine_tuned

esm_fine_tuned = get_model()
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

tokenizer = EsmTokenizer(**tokenizer_config)
torch.cuda.empty_cache()

print("done loading CovFit and tokenizer!")

#########################################################
#               LOADING PLT                             #
#########################################################
parser = argparse.ArgumentParser()
# Path params (defaults handled in main.sh usually, but keeping safe defaults here)
parser.add_argument("--data-dir", type=str, required=True, help="Path to .a2m or .parquet file")
parser.add_argument("--esm2-weight", type=str, required=True, help="Path to ESM2 weights .pt file")
parser.add_argument("--output-dir", type=str, default="results", help="Directory for checkpoints/logs")

# Model params
parser.add_argument("--num-layers", type=int, default=6, help="Total layers in pLM")
parser.add_argument("--d-model", type=int, default=320)
parser.add_argument("--d-hidden", type=int, default=3200, help="Latent dim per layer")

# Training params
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--k", type=int, default=16)
parser.add_argument("--auxk", type=int, default=32)
parser.add_argument("--dead-steps-threshold", type=int, default=10000)
parser.add_argument("--max-epochs", type=int, default=1)
parser.add_argument("--num-devices", type=int, default=1)
parser.add_argument("--wandb-project", type=str, default="ESM-CLT")

plt_data_path = "../../data"
arg_dict = {
    "--data-dir": os.path.join(plt_data_path, "pls_data.parquet"),
    "--esm2-weight":"./covfit_stuff/models/covfit_esm_statedict.pt",
    "--output-dir": "./covfit_stuff/PLT_test",
    "--num-layers": esm_config.num_hidden_layers,
    "--d-model": esm_config.hidden_size, # d_model of ESM
    "--d-hidden": 10 * esm_config.hidden_size, # latent dim of PLT
    "--batch-size": 15, 
}
args = parser.parse_args([str(x) for (k,v) in arg_dict.items() for x in (k,v)])
print("Cross-layer transcoder training params:")
for (k,v) in args.__dict__.items():
    print("%-30s:\t\t%-30s"%(str(k), str(v)))

model = PLTLightningModule(args).to(device)
print("instatiated model!")

run_name = f"PLT_L{args.num_layers}_H{args.d_hidden}"
run_output_dir = os.path.join(args.output_dir, run_name)
os.makedirs(run_output_dir, exist_ok=True)

data_module = SequenceDataModule(args.data_dir, args.batch_size)
# Checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(run_output_dir, "checkpoints"),
    filename="clt-{step}-{val/loss:.2f}",
    save_top_k=2,
    monitor="val/loss", 
    mode="min",
    save_last=True
)
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu",
    devices=args.num_devices,
    # logger=wandb_logger,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
    val_check_interval=100, 
    limit_val_batches=10,
    log_every_n_steps=1 
)
print("validating model before training")
trainer.validate(model, data_module)
torch.cuda.empty_cache()
print("beginning training!")
trainer.fit(model, data_module)
