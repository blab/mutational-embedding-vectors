import torch
import einops
from transformer_lens import (
    HookedTransformerConfig
)

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
    
