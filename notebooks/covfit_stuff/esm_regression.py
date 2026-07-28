"""
ESM regression model for CoVFit project.
Contains the custom EsmForRegression model with multitask learning capabilities.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig, PreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model
from typing import Optional, Tuple
import os
import warnings

from .config import ModelConfig


class EsmForRegression(PreTrainedModel):
    """
    ESM model for regression tasks with multitask learning support.
    
    This model extends the ESM protein language model with task-specific
    regression heads for predicting viral fitness and antibody escape.
    """
    
    config_class = EsmConfig
    _supports_gradient_checkpointing = True  # Enable gradient checkpointing support
    
    def __init__(self, config, n_targets, intermediate_dim=256, dropout_rate=0.5):
        super(EsmForRegression, self).__init__(config)
        self.esm = EsmModel(config)  # Same as original code
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, n_targets)
        )
    
    def _set_gradient_checkpointing(self, module, value=False):
        """
        Enable or disable gradient checkpointing for the ESM model.
        """
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
    
    def weighted_mse_loss(self, inputs, targets, weights):
        diff = inputs - targets
        diff_squared = diff ** 2
        weighted_diff_squared = diff_squared * weights
        loss = weighted_diff_squared.mean()
        return loss

    def forward(self, input_ids, attention_mask=None, labels=None, weights=None, **kwargs):
        outputs = self.esm(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        regression_output = self.regressor(pooled_output)

        if weights is None:
            weights = torch.ones_like(regression_output)

        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)
            masked_labels = labels[mask]
            masked_regression_output = regression_output[mask]
            masked_weights = weights[mask]
            loss = self.weighted_mse_loss(masked_regression_output, masked_labels, masked_weights)

        return SequenceClassifierOutput(
            loss=loss,
            logits=regression_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _load_da_model_safely(model_config: ModelConfig):
    """
    Safely load DA model from HuggingFace Hub or local path with fallback options.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Tuple of (config, model) or None if all attempts fail
    """
    print(f"Attempting to load DA model: {model_config.da_model_name}")
    
    # Try HuggingFace Hub first (if enabled)
    if model_config.use_pretrained_da_model:
        try:
            print(f"Loading from HuggingFace Hub: {model_config.hf_da_model_name}")
            esm_config = EsmConfig.from_pretrained(model_config.hf_da_model_name)
            esm_model = AutoModel.from_pretrained(model_config.hf_da_model_name)
            print("Successfully loaded DA model from HuggingFace Hub")
            return esm_config, esm_model
        except Exception as e:
            print(f"Failed to load from HuggingFace Hub: {e}")
    
    # Try local path if specified
    if model_config.local_da_model_path and os.path.exists(model_config.local_da_model_path):
        try:
            print(f"Loading from local path: {model_config.local_da_model_path}")
            esm_config = EsmConfig.from_pretrained(model_config.local_da_model_path)
            esm_model = AutoModel.from_pretrained(model_config.local_da_model_path)
            print("Successfully loaded DA model from local path")
            return esm_config, esm_model
        except Exception as e:
            print(f"Failed to load from local path: {e}")
    
    # Fallback to base ESM model
    print(f"Falling back to base ESM model: {model_config.model_name}")
    try:
        esm_config = EsmConfig.from_pretrained(model_config.model_name)
        esm_model = AutoModel.from_pretrained(model_config.model_name)
        print("Successfully loaded base ESM model as fallback")
        return esm_config, esm_model
    except Exception as e:
        print(f"Failed to load base ESM model: {e}")
        raise RuntimeError(f"Could not load any ESM model. Please check your configuration and internet connection.")


def create_model_with_lora(model_config: ModelConfig, n_targets: int) -> Tuple[EsmForRegression, nn.Module]:
    """
    Create ESM regression model with LoRA configuration.
    Supports automatic loading from HuggingFace Hub with fallback options.
    
    Args:
        model_config: Model configuration
        n_targets: Number of target tasks
        
    Returns:
        Tuple of (base_model, lora_model)
    """
    try:
        esm_config = EsmConfig.from_pretrained(model_config.da_model_name)
        model = EsmForRegression(esm_config, n_targets)
        model.esm = AutoModel.from_pretrained(model_config.da_model_name)
    except:
        # fallback
        esm_config = EsmConfig.from_pretrained(model_config.model_name)
        model = EsmForRegression(esm_config, n_targets)
        model.esm = AutoModel.from_pretrained(model_config.model_name)
    
    # LoRA Configuration
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        target_modules=["key", "query", "value", "dense"],
        lora_dropout=0.05,
        bias="lora_only",
        modules_to_save=["regressor"]
    )
    lora_model = get_peft_model(model, lora_config)
    
    return model, lora_model


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model to analyze
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def load_model_for_inference(model_path: str, model_config: ModelConfig, n_targets: int) -> nn.Module:
    """
    Load a saved model for inference.
    
    Args:
        model_path: Path to the saved model state dict
        model_config: Model configuration
        n_targets: Number of target tasks
        
    Returns:
        Loaded model ready for inference
    """
    esm_config = EsmConfig.from_pretrained(model_config.da_model_name)
    model = EsmForRegression(esm_config, n_targets)
    model.esm = AutoModel.from_pretrained(model_config.da_model_name)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        target_modules=["key", "query", "value", "dense"],
        lora_dropout=0.05,
        bias="lora_only",
        modules_to_save=["regressor"]
    )
    lora_model = get_peft_model(model, lora_config)
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Fix state_dict inconsistencies
    # Remove unnecessary keys
    keys_to_remove = []
    for key in state_dict.keys():
        if 'contact_head' in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del state_dict[key]
    
    # Load with strict=False for missing keys
    lora_model.load_state_dict(state_dict, strict=False)
    
    lora_model.eval()
    return lora_model


def save_model(model: nn.Module, save_path: str) -> None:
    """
    Save model state dict to file.
    
    Args:
        model: Model to save
        save_path: Path to save the model
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def get_model_predictions(model: nn.Module, dataloader, device: str = 'cuda') -> torch.Tensor:
    """
    Get model predictions for a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing sequences
        device: Device to run inference on
        
    Returns:
        Tensor containing predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_masks'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(outputs.logits.cpu())
    
    return torch.cat(predictions, dim=0)