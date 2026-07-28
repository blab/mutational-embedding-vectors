import torch
from torch import Tensor, nn
import sys
import einops
from jaxtyping import Bool, Float, Int
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from transformers import (
    EsmForMaskedLM, 
    EsmConfig,
    PretrainedConfig, 
    EsmTokenizer, 
    DataCollatorForLanguageModeling, 
    Trainer
)
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
from typing import List, Union, Optional, Callable, Sequence
sys.path.append("../config")
import experiment_config
from covfit_stuff.esm_regression import load_model_for_inference, get_model_predictions, EsmForRegression
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import functools

device = experiment_config.device

def get_hooked_esm_config(esm_cfg, context_len, **kwargs):
    """
    Get hooked transformer config from ESM-2 config
    
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py#L285
    d_model = d_head * n_heads for all ESM models, so d_head = d_model // n_heads
    """
    hooked_esm_config = HookedTransformerConfig(
        n_layers=esm_cfg.num_hidden_layers,
        d_model=esm_cfg.hidden_size,
        d_head=esm_cfg.hidden_size // esm_cfg.num_attention_heads,
        n_heads=esm_cfg.num_attention_heads,
        d_mlp=esm_cfg.intermediate_size,
        d_vocab=esm_cfg.vocab_size,
        n_ctx=context_len,
        act_fn=esm_cfg.hidden_act,
        normalization_type="LN",
        positional_embedding_type="rotary",
        attention_dir="bidirectional",
        post_embedding_ln=False,
        tokenizer_name=esm_cfg.model_name,
        d_vocab_out=esm_cfg.hidden_size,
        eps=esm_cfg.layer_norm_eps,
        **kwargs
    )
    return hooked_esm_config

def get_logits_hooked_esm(hooked_esm_final_layer, ESM2_lm_head):
    """
    get final logits of hooked esm (kinda hacky)
    hooked_esm_final_layer: output of model after all layers
    ESM2_lm_head: ESM-2 Language modeling head

    See below for more details:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py#L724
    """
    with torch.no_grad():
        output_logits = ESM2_lm_head(hooked_esm_final_layer)
    return output_logits

def rotary_embeddings(inv_freq, cfg, device="cuda"):
    """
    Helper function to create rotary embedding matrices from inv_freq from hugging face ESM-2 state dict
    
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py#L80
    """
    t = torch.arange(cfg.n_ctx).to(inv_freq.device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(inv_freq.device)
    cos_cached = emb.cos()
    sin_cached = emb.sin()
    
    return cos_cached.to(device), sin_cached.to(device)

def get_hooked_state_dict(hf_esm_state_dict, cfg, device="cuda"):
    """
    hugging face ESM-2 state dict -> hooked transformer state dict

    hf_esm_state_dict: state dict of ESM model (from hugging face)
    cfg: hooked Transformer config
    device: "cpu" or "cuda"
    """
    old_state_dict_keys = hf_esm_state_dict.keys()
    new_state_dict = {}

    old_to_new_weights = {
        "attention.self.query.weight":"attn.W_Q",
        "attention.self.key.weight":"attn.W_K",
        "attention.self.value.weight":"attn.W_V",
        "attention.output.dense.weight":"attn.W_O", 
    }
    old_to_new_bias = {
        "attention.self.query.bias":"attn.b_Q",
        "attention.self.key.bias":"attn.b_K",
        "attention.self.value.bias":"attn.b_V",
        "attention.output.dense.bias":"attn.b_O"
    }
    old_to_new_mlp = {
        "intermediate.dense.weight":"mlp.W_in",
        "intermediate.dense.bias":"mlp.b_in",
        "output.dense.weight":"mlp.W_out",
        "output.dense.bias":"mlp.b_out",
    }
    old_to_new_ln = {
        "attention.LayerNorm.weight":"ln1.w",
        "attention.LayerNorm.bias":"ln1.b",
        "LayerNorm.weight":"ln2.w",
        "LayerNorm.bias":"ln2.b"
    }

    # embedding matrix
    new_state_dict["embed.W_E"] = hf_esm_state_dict["esm.embeddings.word_embeddings.weight"]

    # hacky unembedding matrix is just the identity
    new_state_dict["unembed.W_U"] = torch.eye(cfg.d_model, cfg.d_vocab_out)
    new_state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab_out)
    
    
    for l in range(cfg.n_layers):
        l_keys = [x for x in old_state_dict_keys if f".{l}." in x]
        old_prefix = f"esm.encoder.layer.{l}"
        new_prefix = f"blocks.{l}"

        # attn ignore = -inf
        new_state_dict[f"{new_prefix}.attn.IGNORE"] = torch.tensor(-torch.inf).to(device)
        
        # bidirectional attention, so attention should be looking everywhere
        new_state_dict[f"{new_prefix}.attn.mask"] = torch.full((cfg.n_ctx, cfg.n_ctx), True)

        # rotary embeddings
        cos_cached, sin_cached = rotary_embeddings(hf_esm_state_dict[f"esm.encoder.layer.{l}.attention.self.rotary_embeddings.inv_freq"], cfg, device)
        new_state_dict[f"{new_prefix}.attn.rotary_cos"] = cos_cached
        new_state_dict[f"{new_prefix}.attn.rotary_sin"] = sin_cached
        
        # weights
        for w in old_to_new_weights.keys():
            # weights are arranged [out_features, in_features] = [n_head * d_head, d_model]
            new_weight_name = old_to_new_weights[w]
            if "output" in w:
                # [d_model d_head]
                new_state_dict[f"{new_prefix}.{new_weight_name}"] = einops.rearrange(hf_esm_state_dict[f"{old_prefix}.{w}"], "d_model (n_head d_head) -> n_head d_head d_model", n_head=cfg.n_heads)
            else:
                new_state_dict[f"{new_prefix}.{new_weight_name}"] = einops.rearrange(hf_esm_state_dict[f"{old_prefix}.{w}"], "(n_head d_head) d_model -> n_head d_model d_head", n_head=cfg.n_heads)
            
        #biases
        for b in old_to_new_bias.keys():
            new_bias_name = old_to_new_bias[b]
            if "output" in b:
                new_state_dict[f"{new_prefix}.{new_bias_name}"] = hf_esm_state_dict[f"{old_prefix}.{b}"]
            else:
                new_state_dict[f"{new_prefix}.{new_bias_name}"] = einops.rearrange(hf_esm_state_dict[f"{old_prefix}.{b}"], "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
            
        # mlp 
        for m in old_to_new_mlp.keys():
            # mlp are arranged [out_features, in_features] = [d_mlp, d_model]
            new_mlp_name = old_to_new_mlp[m]
            # mlp weights
            if "weight" in m:
                new_state_dict[f"{new_prefix}.{new_mlp_name}"] = einops.rearrange(hf_esm_state_dict[f"{old_prefix}.{m}"], "out_feats in_feats -> in_feats out_feats")
            # mlp biases
            else:
                new_state_dict[f"{new_prefix}.{new_mlp_name}"] = hf_esm_state_dict[f"{old_prefix}.{m}"]

        # layernorms
        for ln in old_to_new_ln.keys():
            new_ln_name = old_to_new_ln[ln]
            new_state_dict[f"{new_prefix}.{new_ln_name}"] = hf_esm_state_dict[f"{old_prefix}.{ln}"]

        # Final LayerNorm
        new_state_dict["ln_final.w"] = hf_esm_state_dict["esm.encoder.emb_layer_norm_after.weight"]
        new_state_dict["ln_final.b"] = hf_esm_state_dict["esm.encoder.emb_layer_norm_after.bias"]

    return new_state_dict

def get_fairesm_state_dict(hf_esm_state_dict, cfg, device="cuda"):
    """
    hugging face ESM-2 state dict -> hooked transformer state dict

    hf_esm_state_dict: state dict of ESM model (from hugging face)
    cfg: huggingface ESM_CONFIG
    device: "cpu" or "cuda"
    """
    old_state_dict_keys = hf_esm_state_dict.keys()
    new_state_dict = {}

    old_to_new_weights = {
        "attention.self.query.weight":"self_attn.q_proj.weight",
        "attention.self.key.weight":"self_attn.k_proj.weight",
        "attention.self.value.weight":"self_attn.v_proj.weight",
        "attention.output.dense.weight":"self_attn.out_proj.weight", 
    }
    old_to_new_bias = {
        "attention.self.query.bias":"self_attn.q_proj.bias",
        "attention.self.key.bias":"self_attn.k_proj.bias",
        "attention.self.value.bias":"self_attn.v_proj.bias",
        "attention.output.dense.bias":"self_attn.out_proj.bias"
    }
    old_to_new_mlp = {
        "intermediate.dense.weight":"fc1.weight",
        "intermediate.dense.bias":"fc1.bias",
        "output.dense.weight":"fc2.weight",
        "output.dense.bias":"fc2.bias",
    }
    old_to_new_ln = {
        "attention.LayerNorm.weight":"self_attn_layer_norm.weight",
        "attention.LayerNorm.bias":"self_attn_layer_norm.bias",
        "LayerNorm.weight":"final_layer_norm.weight",
        "LayerNorm.bias":"final_layer_norm.bias"
    }

    # embedding matrix
    new_state_dict["embed_tokens.weight"] = hf_esm_state_dict["esm.embeddings.word_embeddings.weight"]
    
    
    for l in range(cfg.num_hidden_layers):
        l_keys = [x for x in old_state_dict_keys if f".{l}." in x]
        old_prefix = f"esm.encoder.layer.{l}"
        new_prefix = f"layers.{l}"

        # rotary embeddings
        new_state_dict[f"{new_prefix}.self_attn.rot_emb.inv_freq"] = hf_esm_state_dict[f"esm.encoder.layer.{l}.attention.self.rotary_embeddings.inv_freq"]
        
        # weights
        for w in old_to_new_weights.keys():
            # weights are arranged [out_features, in_features] = [n_head * d_head, d_model]
            new_weight_name = old_to_new_weights[w]
            new_state_dict[f"{new_prefix}.{new_weight_name}"] = hf_esm_state_dict[f"{old_prefix}.{w}"]
            
        #biases
        for b in old_to_new_bias.keys():
            new_bias_name = old_to_new_bias[b]
            new_state_dict[f"{new_prefix}.{new_bias_name}"] = hf_esm_state_dict[f"{old_prefix}.{b}"]
            
        # mlp 
        for m in old_to_new_mlp.keys():
            # mlp are arranged [out_features, in_features] = [d_mlp, d_model]
            new_mlp_name = old_to_new_mlp[m]
            new_state_dict[f"{new_prefix}.{new_mlp_name}"] = hf_esm_state_dict[f"{old_prefix}.{m}"]

        # layernorms
        for ln in old_to_new_ln.keys():
            new_ln_name = old_to_new_ln[ln]
            new_state_dict[f"{new_prefix}.{new_ln_name}"] = hf_esm_state_dict[f"{old_prefix}.{ln}"]

        # Final LayerNorm
        new_state_dict["emb_layer_norm_after.weight"] = hf_esm_state_dict["esm.encoder.emb_layer_norm_after.weight"]
        new_state_dict["emb_layer_norm_after.bias"] = hf_esm_state_dict["esm.encoder.emb_layer_norm_after.bias"]

    return new_state_dict

# add padding mask to model
def add_perma_hooks_to_mask_pad_tokens(
    model: HookedTransformer, pad_token: int
) -> HookedTransformer:
    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(tokens: Float[Tensor, "batch seq"], hook: HookPoint) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model

def get_model(
    TOK_DIR = "./covfit_stuff/Tokenizer",
    CONF_DIR = "./covfit_stuff/Config",
    TASK_IDS_FILE = "./covfit_stuff/task_id_dict.pt",
    FOLD_ID = 0,
    N_TARGETS = 1565,
    MODEL_PATH = f"./covfit_stuff/models/covfit_model_20241007_0.ckpt",
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

##################################################
#               Helper Functions                 #
##################################################
# Activation patching
def patch_head_vector(
    clean_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index: int,
    corrupted_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    Patches the output of a given head (before it's added to the residual stream) at every sequence
    position, using the value from the corrupted cache.
    """
    clean_head_vector[:,:,head_index,:] = corrupted_cache[hook.name][:,:,head_index,:]
    return clean_head_vector

def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer,
    logit_id: int,
    receiver_input: str, # one of ["k", "q", "v", "z", "pattern"]
    clean_tokens: Float[Tensor, "batch pos"],
    corrupted_cache: ActivationCache,
    patching_metric: Callable,
    get_logit_hooked: Callable
) -> Float[Tensor, "layer head"]:
    """
    Returns an array of results of patching at all positions for each head in each layer, using the
    value from the clean cache. The results are calculated using the patching_metric function, which
    should be called on the model's logit output.
    """
    model.reset_hooks(including_permanent=False)

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    results = torch.zeros((num_layers, num_heads)).to(device)

    fancy_progress_bar = tqdm(total = num_layers * num_heads)
    for layer in range(num_layers):
        for head in range(num_heads):
            partial_hook_f = functools.partial(patch_head_vector, head_index=head, corrupted_cache=corrupted_cache)

            if receiver_input == "pattern":
                patched_component_name = [utils.get_act_name("q", layer), utils.get_act_name("k", layer)]
                patched_output = model.run_with_hooks(clean_tokens, fwd_hooks=[(p_name, partial_hook_f) for p_name in patched_component_name])
            else:
                patched_component_name = utils.get_act_name(receiver_input, layer)
                patched_output = model.run_with_hooks(clean_tokens, fwd_hooks=[(patched_component_name, partial_hook_f)])
            
            patched_output_logits = get_logit_hooked(patched_output, tok_id=logit_id)
            results[layer, head] = patching_metric(patched_output_logits)
            torch.cuda.empty_cache()

            fancy_progress_bar.update()
    fancy_progress_bar.close()
    
    return results

def patch_mlp_vector(
    clean_mlp_vector: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    corrupted_cache: ActivationCache
) -> Float[Tensor, "batch pos d_mlp"]:
    """
    Patches the output of a given mlp (before it's added to the residual stream) at every sequence
    position, using the value from the corrupted cache.
    """
    clean_mlp_vector[...] = corrupted_cache[hook.name][...]
    return clean_mlp_vector

def get_act_patch_mlp_all_pos(
    model: HookedTransformer,
    logit_id: int,
    clean_tokens: Float[Tensor, "batch pos"],
    corrupted_cache: ActivationCache,
    patching_metric: Callable,
    get_logit_hooked: Callable
) -> Float[Tensor, "layer 1"]:
    """
    Returns an array of results of patching at all positions for each head in each layer, using the
    value from the clean cache. The results are calculated using the patching_metric function, which
    should be called on the model's logit output.
    """
    
    model.reset_hooks(including_permanent=False)
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    results = torch.zeros((num_layers, 1)).to(device)

    mlp_names = [f"blocks.{layer}.hook_mlp_out" for layer in range(num_layers)]

    with tqdm(total = num_layers) as fancy_progress_bar:
        partial_hook_f = functools.partial(patch_mlp_vector, corrupted_cache=corrupted_cache)
        for layer in range(num_layers):
            model.reset_hooks(including_permanent=False)
            patched_component_name = mlp_names[layer]
            patched_output = model.run_with_hooks(clean_tokens, fwd_hooks=[(patched_component_name, partial_hook_f)])
            
            patched_output_logits = get_logit_hooked(patched_output, tok_id=logit_id)
            results[layer, :] = patching_metric(patched_output_logits)
            torch.cuda.empty_cache()

            fancy_progress_bar.update()
    
    return results

def abs_path_patch_head_input(
    orig_head_vector: Float[Tensor, "batch pos n_head d_head"],
    hook: HookPoint,
    new_comp: Float[Tensor, "batch pos d_head"], # changed_head_vector at head_idx
    head_idx: int
):
    orig_head_vector[:,:,head_idx,:] = new_comp
    return orig_head_vector


def abs_path_patch_mlp_input(
    orig_mlp_vector: Float[Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    new_comp: Float[Tensor, "batch pos d_mlp"], # changed_mlp_vector
):
    orig_mlp_vector[...] = new_comp[...]
    return orig_mlp_vector

def patch_or_freeze_head_vectors(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int], # [layer, head]
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them to their
    values in orig_cache), except for head_to_patch (if it's in this layer) which we patch with the
    value from new_cache.

    head_to_patch: tuple of (layer, head)
    """
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    """
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    """
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation

    
##################################################
#               Interp Functions                 #
##################################################
# direct to resid stream
def path_patch_head_absolute_direct(
    model,
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache,
    logit_id: int,
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    path_path_results = torch.zeros(num_layers, num_heads)

    z_name_list = [f"blocks.{layer}.attn.hook_z" for layer in range(num_layers)]
    names_filter = ["blocks.32.hook_resid_post", *z_name_list]
    _, orig_cache = model.run_with_cache(orig_dataset, names_filter=names_filter)

    esm_output = orig_cache["blocks.32.hook_resid_post"] # [batch pos d_model]
    final_ln = model.ln_final
    
    torch.cuda.empty_cache()
    
    with tqdm(total = num_layers * num_heads) as fancy_progress_bar:
        for layer in range(num_layers):
            W_O_layer = model.W_O[layer] # [n_head d_head d_model]
            b_O_layer = model.b_O[layer]
 
            for head in range(num_heads):
                head_name = f"blocks.{layer}.attn.hook_z"
                layer_head_output_orig = einops.einsum(orig_cache[head_name], W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]

                layer_head_z_new = orig_cache[head_name].detach().clone() # [batch pos n_head d_head]
                layer_head_z_new[:,:,head,:] = new_cache[head_name][:,:,head,:]
                layer_head_output_new = einops.einsum(layer_head_z_new, W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]

                path_patched_logit = final_ln(esm_output - layer_head_output_orig + layer_head_output_new)
                # step 3, compute final logits
                path_patched_logits = get_logit_hooked(path_patched_logit, logit_id)
                torch.cuda.empty_cache()
        
                path_path_results[layer, head] = patching_metric(path_patched_logits)
                
                fancy_progress_bar.update()

    return path_path_results


def path_patch_mlp_absolute_direct(
    model,
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache,
    logit_id: int,
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    path_patch_results = torch.zeros(num_layers).to(device)

    mlp_names_filter = [f"blocks.{layer}.hook_mlp_out" for layer in range(model.cfg.n_layers)]
    names_filter = ["blocks.32.hook_resid_post", *mlp_names_filter]
    _, orig_cache = model.run_with_cache(orig_dataset, names_filter=names_filter)

    esm_output = orig_cache["blocks.32.hook_resid_post"] # [batch pos d_model]
    final_ln = model.ln_final
    
    torch.cuda.empty_cache()
    
    with tqdm(total = num_layers * 1) as fancy_progress_bar:
        for layer in range(num_layers):
            mlp_name = mlp_names_filter[layer] 
            mlp_orig_output = orig_cache[mlp_name] # [batch pos d_model]
            mlp_new_output = new_cache[mlp_name] # [batch pos d_model]

            path_patched_logit = final_ln(esm_output - mlp_orig_output + mlp_new_output) # [batch pos d_model]
            # step 3, compute final logits
            path_patched_logits = get_logit_hooked(path_patched_logit, tok_id=logit_id) # [batch]
            path_patch_results[layer] = patching_metric(path_patched_logits)
            torch.cuda.empty_cache()
            fancy_progress_bar.update()

    return path_patch_results

def abs_path_patch_receiver_heads(
    model: HookedTransformer,
    receiver_heads: list[tuple[int, int]], # (layer,head)
    receiver_input: str, #one of {"q", "k", "v"}
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache: ActivationCache, # cached attn. z
    orig_cache: ActivationCache, # cached attn. z
    logit_id: int
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    path_patch_results = torch.zeros(num_layers, num_heads)

    # inputs to receiver head are output of prev layer
    receiver_layer_inputs = [x[0] - 1 for x in receiver_heads]

    # receiver head layers
    receiver_layers = [x[0] for x in receiver_heads]
    
    layer_head_lns = [model.blocks[layer].ln1 for layer in receiver_layers]
    resid_post_names = [f"blocks.{layer}.hook_resid_post" for layer in receiver_layer_inputs]
    _, resid_post_cache = model.run_with_cache(orig_dataset, names_filter = resid_post_names)
    torch.cuda.empty_cache()

    W_receiver = {"q": model.W_Q, "k": model.W_K, "v": model.W_V}[receiver_input] # [n_layer n_head d_model d_head]
    b_receiver = {"q": model.b_Q, "k": model.b_K, "v": model.b_V}[receiver_input] # [n_layer n_head d_head]
    
    with tqdm(total = max(receiver_layers) * num_heads) as fancy_progress_bar:
        for layer in range(max(receiver_layers)):
            W_O_layer = model.W_O[layer] # [n_head d_head d_model]
            b_O_layer = model.b_O[layer]
 
            for head in range(num_heads):
                model.reset_hooks()
                head_name = f"blocks.{layer}.attn.hook_z"

                # STEP 2
                # new_resid_input= old_resid_input - old_head_output + new_head_output
                # new_head_output
                new_head_z = orig_cache[head_name].detach().clone() # [batch pos n_head d_head]
                new_head_z[:,:, head, :] = new_cache[head_name][:,:,head,:] # [batch pos n_head d_head]
                new_head_output = einops.einsum(new_head_z, W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]
                # old_head_output
                orig_head_output = einops.einsum(orig_cache[head_name], W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]

                # add hooks (compute what the new component would've been using new_resid_input and patch that in using abs_path_patch_head_input)
                for r_ln, r_post_name, (rl, rh) in zip(layer_head_lns, resid_post_names, receiver_heads):
                    new_resid_input = r_ln(resid_post_cache[r_post_name] - orig_head_output + new_head_output)
                    new_comp = einops.einsum(new_resid_input, W_receiver[rl], "batch pos d_model, n_head d_model d_head -> batch pos n_head d_head") + b_receiver[rl] # [batch pos n_head d_head]
                    
                    model.add_hook(
                        utils.get_act_name(receiver_input, rl),
                        functools.partial(abs_path_patch_head_input, new_comp=new_comp[:,:,rh,:], head_idx=rh), 
                        dir="fwd"
                    )
                
                path_patched_logits = model(orig_dataset) # [batch pos d_model]
                # step 3, compute final logits
                path_patched_logits = get_logit_hooked(path_patched_logits, logit_id)
                torch.cuda.empty_cache()
        
                path_patch_results[layer, head] = patching_metric(path_patched_logits)
                
                fancy_progress_bar.update()
        model.reset_hooks()
    return path_patch_results


def abs_path_patch_mlp_to_receiver_heads(
    model: HookedTransformer,
    receiver_heads: list[tuple[int, int]], # (layer,head)
    receiver_input: str, #one of {"q", "k", "v"}
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache: ActivationCache, # cached mlp
    orig_cache: ActivationCache, # cached mlp
    logit_id: int
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    path_patch_results = torch.zeros(num_layers)

    # inputs to receiver head are output of prev layer
    receiver_layer_inputs = [x[0] - 1 for x in receiver_heads]

    # receiver head layers
    receiver_layers = [x[0] for x in receiver_heads]
    
    layer_head_lns = [model.blocks[layer].ln1 for layer in receiver_layers]
    resid_post_names = [f"blocks.{layer}.hook_resid_post" for layer in receiver_layer_inputs]
    _, resid_post_cache = model.run_with_cache(orig_dataset, names_filter = resid_post_names)
    torch.cuda.empty_cache()

    W_receiver = {"q": model.W_Q, "k": model.W_K, "v": model.W_V}[receiver_input]
    b_receiver = {"q": model.b_Q, "k": model.b_K, "v": model.b_V}[receiver_input]
    
    with tqdm(total = max(receiver_layers)) as fancy_progress_bar:
        for layer in range(max(receiver_layers)):
            model.reset_hooks()
            mlp_layer_name = f"blocks.{layer}.hook_mlp_out"

            # STEP 2
            # new_resid_input= old_resid_input - old_head_output + new_head_output
            # new_head_output
            new_mlp_output = new_cache[mlp_layer_name]
            # old_head_output
            orig_mlp_output = orig_cache[mlp_layer_name]

            # add hooks (compute what the new component would've been using new_resid_input and patch that in using abs_path_patch_head_input)
            for r_ln, r_post_name, (rl, rh) in zip(layer_head_lns, resid_post_names, receiver_heads):
                new_resid_input = r_ln(resid_post_cache[r_post_name] - orig_mlp_output + new_mlp_output)
                new_comp = einops.einsum(new_resid_input, W_receiver[rl], "batch pos d_model, n_head d_model d_head -> batch pos n_head d_head") + b_receiver[rl] # [batch pos n_head d_head]
                
                model.add_hook(
                    utils.get_act_name(receiver_input, rl), 
                    functools.partial(abs_path_patch_head_input, new_comp=new_comp[:,:,rh,:], head_idx=rh), 
                    dir="fwd"
                )
            
            path_patched_logits = model(orig_dataset) # [batch pos d_model]
            # step 3, compute final logits
            path_patched_logits = get_logit_hooked(path_patched_logits, logit_id)
            torch.cuda.empty_cache()
    
            path_patch_results[layer] = patching_metric(path_patched_logits)
            
            fancy_progress_bar.update()
        model.reset_hooks()
    return path_patch_results


def abs_path_patch_to_mlp(
    model: HookedTransformer,
    receiver_mlps: list[int], # (layers)
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache: ActivationCache, # cached attn. z
    orig_cache: ActivationCache, # cached attn. z
    logit_id: int
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    path_patch_results = torch.zeros(num_layers, num_heads)
    
    layer_head_lns = [model.blocks[layer].ln2 for layer in receiver_mlps]
    resid_mid_names = [f"blocks.{layer}.hook_resid_mid" for layer in receiver_mlps]
    _, resid_mid_cache = model.run_with_cache(orig_dataset, names_filter = resid_mid_names)
    torch.cuda.empty_cache()

    W_mlp_in = model.W_in # [n_layer d_model d_mlp]
    b_mlp_in = model.b_in # # [n_layer d_mlp]
    
    with tqdm(total = (max(receiver_mlps) + 1) * num_heads) as fancy_progress_bar:
        for layer in range(max(receiver_mlps) + 1):
            W_O_layer = model.W_O[layer] # [n_head d_head d_model]
            b_O_layer = model.b_O[layer]
 
            for head in range(num_heads):
                model.reset_hooks()
                head_name = f"blocks.{layer}.attn.hook_z"

                # STEP 2
                # new_resid_input= old_resid_input - old_head_output + new_head_output
                # new_head_output
                new_head_z = orig_cache[head_name].detach().clone() # [batch pos n_head d_head]
                new_head_z[:,:, head, :] = new_cache[head_name][:,:,head,:] # [batch pos n_head d_head]
                new_head_output = einops.einsum(new_head_z, W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]
                # old_head_output
                orig_head_output = einops.einsum(orig_cache[head_name], W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]

                # add hooks (compute what the new component would've been using new_resid_input and patch that in using abs_path_patch_head_input)
                for r_ln, r_post_name, rl in zip(layer_head_lns, resid_mid_names, receiver_mlps):
                    # only patch in things from earlier layers (including current layer since MHA comes before MLP)
                    if rl >= layer:
                        receiver_mlp_name = f"blocks.{rl}.mlp.hook_pre"
                        new_resid_input = r_ln(resid_mid_cache[r_post_name] - orig_head_output + new_head_output)
                        new_comp = einops.einsum(new_resid_input, W_mlp_in[rl], "batch pos d_model, d_model d_mlp -> batch pos d_mlp") + b_mlp_in[rl] # [batch pos n_head d_head]
                        
                        model.add_hook(
                            receiver_mlp_name, 
                            functools.partial(abs_path_patch_mlp_input, new_comp=new_comp), 
                            dir="fwd"
                        )
                
                path_patched_logits = model(orig_dataset) # [batch pos d_model]
                # step 3, compute final logits
                path_patched_logits = get_logit_hooked(path_patched_logits, logit_id)
                torch.cuda.empty_cache()
        
                path_patch_results[layer, head] = patching_metric(path_patched_logits)
                
                fancy_progress_bar.update()
        model.reset_hooks()
    return path_patch_results


def abs_path_patch_to_mlp_make_31_eq_0(
    model: HookedTransformer,
    receiver_mlps: list[int], # (layers)
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache: ActivationCache, # cached attn. z
    orig_cache: ActivationCache, # cached attn. z
    logit_id: int
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    path_patch_results = torch.zeros(num_layers, num_heads)
    
    layer_head_lns = [model.blocks[layer].ln2 for layer in receiver_mlps]
    resid_mid_names = [f"blocks.{layer}.hook_resid_mid" for layer in receiver_mlps]
    _, resid_mid_cache = model.run_with_cache(orig_dataset, names_filter = resid_mid_names)
    torch.cuda.empty_cache()

    W_mlp_in = model.W_in # [n_layer d_model d_mlp]
    b_mlp_in = model.b_in # # [n_layer d_mlp]

    batch_dim, pos_dim = orig_dataset.shape
    head_dim = model.cfg.d_head
    
    with tqdm(total = (max(receiver_mlps) + 1) * num_heads) as fancy_progress_bar:
        for layer in range(max(receiver_mlps) + 1):
            W_O_layer = model.W_O[layer] # [n_head d_head d_model]
            b_O_layer = model.b_O[layer]
 
            for head in range(num_heads):
                if head == 7 and layer == 31:
                    continue
                model.reset_hooks()
                # MAKE 31.7 == 0
                # ---------------------
                model.add_hook(
                    "blocks.31.attn.hook_z", 
                    functools.partial(abs_path_patch_head_input, new_comp=torch.zeros(batch_dim, pos_dim, head_dim).to(device), head_idx=7), 
                    dir="fwd"
                )
                # ---------------------
                
                head_name = f"blocks.{layer}.attn.hook_z"

                # STEP 2
                # new_resid_input= old_resid_input - old_head_output + new_head_output
                # new_head_output
                new_head_z = orig_cache[head_name].detach().clone() # [batch pos n_head d_head]
                new_head_z[:,:, head, :] = new_cache[head_name][:,:,head,:] # [batch pos n_head d_head]
                new_head_output = einops.einsum(new_head_z, W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]
                # old_head_output
                orig_head_output = einops.einsum(orig_cache[head_name], W_O_layer, "batch pos n_head d_head, n_head d_head d_model -> batch pos d_model") + b_O_layer # [batch pos d_model]

                # add hooks (compute what the new component would've been using new_resid_input and patch that in using abs_path_patch_head_input)
                for r_ln, r_post_name, rl in zip(layer_head_lns, resid_mid_names, receiver_mlps):
                    # only patch in things from earlier layers (including current layer since MHA comes before MLP)
                    if rl >= layer:
                        receiver_mlp_name = f"blocks.{rl}.mlp.hook_pre"
                        new_resid_input = r_ln(resid_mid_cache[r_post_name] - orig_head_output + new_head_output)
                        new_comp = einops.einsum(new_resid_input, W_mlp_in[rl], "batch pos d_model, d_model d_mlp -> batch pos d_mlp") + b_mlp_in[rl] # [batch pos n_head d_head]
                        
                        model.add_hook(
                            receiver_mlp_name, 
                            functools.partial(abs_path_patch_mlp_input, new_comp=new_comp), 
                            dir="fwd"
                        )
                
                path_patched_logits = model(orig_dataset) # [batch pos d_model]
                # step 3, compute final logits
                path_patched_logits = get_logit_hooked(path_patched_logits, logit_id)
                torch.cuda.empty_cache()
        
                path_patch_results[layer, head] = patching_metric(path_patched_logits)
                
                fancy_progress_bar.update()
        model.reset_hooks()
    return path_patch_results


def abs_direc_path_patch_mlp_to_mlp(
    model: HookedTransformer,
    receiver_mlps: list[int], # (layers)
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache: ActivationCache, # cached attn. z
    orig_cache: ActivationCache, # cached attn. z
    logit_id: int
) -> Float[Tensor, "num_layer num_head"]:

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    path_patch_results = torch.zeros(num_layers)
    
    layer_head_lns = [model.blocks[layer].ln2 for layer in receiver_mlps]
    
    resid_mid_names = [f"blocks.{layer}.hook_resid_mid" for layer in receiver_mlps]
    W_mlp_in = model.W_in # [n_layer d_model d_mlp]
    b_mlp_in = model.b_in # # [n_layer d_mlp]
    
    with tqdm(total = (max(receiver_mlps)) * 1) as fancy_progress_bar:
        for layer in range(max(receiver_mlps)):
            model.reset_hooks()

            mlp_layer_name = f"blocks.{layer}.hook_resid_mid"
            ln2_layer_f = model.blocks[layer].ln2
            mlp_layer_f = model.blocks[layer].mlp

            # STEP 2
            # new_resid_input= old_resid_input - old_head_output + new_head_output
            # new_head_output
            new_mlp_output = mlp_layer_f(ln2_layer_f(new_cache[mlp_layer_name]))
            torch.cuda.empty_cache()
            
            # old_head_output
            orig_mlp_output = mlp_layer_f(ln2_layer_f(orig_cache[mlp_layer_name])) 
            torch.cuda.empty_cache()

            # add hooks (compute what the new component would've been using new_resid_input and patch that in using abs_path_patch_head_input)
            for r_ln, r_post_name, rl in zip(layer_head_lns, resid_mid_names, receiver_mlps):
                # only patch in things from strictly earlier layers
                if rl > layer:
                    receiver_mlp_name = f"blocks.{rl}.mlp.hook_pre"
                    new_resid_input = r_ln(orig_cache[r_post_name] - orig_mlp_output + new_mlp_output)
                    new_comp = einops.einsum(new_resid_input, W_mlp_in[rl], "batch pos d_model, d_model d_mlp -> batch pos d_mlp") + b_mlp_in[rl] # [batch pos n_head d_head]
                    
                    model.add_hook(
                        receiver_mlp_name, 
                        functools.partial(abs_path_patch_mlp_input, new_comp=new_comp), 
                        dir="fwd"
                    )
                
            path_patched_logits = model(orig_dataset) # [batch pos d_model]
            # step 3, compute final logits
            path_patched_logits = get_logit_hooked(path_patched_logits, logit_id)
            del new_mlp_output
            del orig_mlp_output
            del new_resid_input
            torch.cuda.empty_cache()
    
            path_patch_results[layer] = patching_metric(path_patched_logits)
            
            fancy_progress_bar.update()
        model.reset_hooks()
    return path_patch_results

def get_path_patch_head_to_heads(
    model: HookedTransformer,
    receiver_heads: list[tuple[int, int]], # (layer,head)
    receiver_input: str,
    new_dataset: Float[Tensor, "batch pos"],
    orig_dataset: Float[Tensor, "batch pos"],
    patching_metric: Callable,
    get_logit_hooked: Callable,
    new_cache: ActivationCache | None,
    orig_cache: ActivationCache | None,
    logit_id: int
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input, for example if
    receiver_input = "v" and receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)], we're doing path
    patching from each head to the value inputs of the S-inhibition heads.

    Returns:
        tensor of metric values for every possible sender head
    """

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads

    results = torch.zeros((num_layers, num_heads)).to(model.cfg.device)
    z_name_filter = lambda name: name.endswith("z")

    receiver_layers = set([x[0] for x in receiver_heads])
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    # step 1, get cached runs
    if new_cache == None:
        _, new_cache = model.run_with_cache(new_dataset, names_filter=z_name_filter)
        torch.cuda.empty_cache()

    if orig_cache == None:
        _, orig_cache = model.run_with_cache(orig_dataset, names_filter=z_name_filter)
        torch.cuda.empty_cache()

    with tqdm(total = num_layers * num_heads) as fancy_progress_bar:
        for layer in range(num_layers):
            for head in range(num_heads):
                model.reset_hooks()
    
                # step 2, run original dataset with non-direct paths frozen
                step2_hook_f = functools.partial(patch_or_freeze_head_vectors, new_cache=new_cache, orig_cache=orig_cache, head_to_patch=(layer, head))
                model.add_hook(z_name_filter, step2_hook_f, dir="fwd")
                _, patching_cache = model.run_with_cache(orig_dataset, names_filter=receiver_hook_names_filter)
                torch.cuda.empty_cache()
    
                # step 3, compute final logits
                model.reset_hooks()
                step3_hook_f = functools.partial(patch_head_input, patched_cache=patching_cache, head_list=receiver_heads)
                path_patched_logits_esm = model.run_with_hooks(orig_dataset, fwd_hooks=[(receiver_hook_names_filter, step3_hook_f)], return_type="logits")
                path_patched_logits = get_logit_hooked(path_patched_logits_esm, logit_id)
                torch.cuda.empty_cache()
    
                results[layer, head] = patching_metric(path_patched_logits)
                fancy_progress_bar.update()
    model.reset_hooks()
    return results