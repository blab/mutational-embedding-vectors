"""
Configuration file for CoVFit training and inference.
Contains all hyperparameters and settings extracted from original training script.
"""

import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Data paths - loaded from environment variables (Docker-friendly defaults)
    dir: str = os.environ.get('BASE_DIR', '/workspace/')
    full_df_name: str = os.environ.get('METADATA_FILE', 'data/raw/metadata.representative.all_countries.with_date.v2.with_seq_231102_wo_variants_before_cutoff.txt')
    antibody_escape_f_name: str = os.environ.get('ANTIBODY_ESCAPE_FILE', 'data/raw/escape_data_mutation.csv')
    dms_fasta_f_name: str = os.environ.get('DMS_FASTA_FILE', 'data/raw/nextclade.peptide.S_rename.fasta')
    
    # Data filtering parameters
    min_country: int = 300  # Countries with <300 genotype-fitness data were removed
    target_strain: str = "D614G"
    negative_clade_list: List[str] = None
    negative_source_list: List[str] = None
    max_ic50: float = 10.0
    max_x_count: int = 5
    max_hyphen_count: int = 30
    
    # Date thresholds
    date_thresh: str = "2019-11-01"  # for loss weighting by variant emergence date
    cutoff: str = "2023-11-01"  # training cutoff date
    
    def __post_init__(self):
        if self.negative_clade_list is None:
            self.negative_clade_list = ["recombinant"]
        if self.negative_source_list is None:
            self.negative_source_list = ['SARS convalescents', 'WT-engineered']
        
        # Convert date strings to pandas datetime
        self.date_thresh = pd.to_datetime(self.date_thresh)
        self.cutoff = pd.to_datetime(self.cutoff)


@dataclass 
class ModelConfig:
    """Model architecture configuration"""
    # Model parameters
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    da_model_name: str = None  # Will be set based on dir or HuggingFace Hub
    max_length: int = 1024  # max token length
    intermediate_dim: int = 256
    dropout_rate: float = 0.5
    
    # HuggingFace Hub integration
    use_pretrained_da_model: bool = True  # Whether to use HuggingFace pretrained DA model
    hf_da_model_name: str = "your-username/covfit-da-model"  # HuggingFace model repository
    local_da_model_path: str = None  # Local path to DA model (fallback)
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["key", "query", "value", "dense"]
        
        # Set DA model name based on configuration
        if self.use_pretrained_da_model and self.da_model_name is None:
            self.da_model_name = self.hf_da_model_name
        elif not self.use_pretrained_da_model and self.local_da_model_path:
            self.da_model_name = self.local_da_model_path


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 20
    weight_decay: float = 0.02
    warmup_ratio: float = 0.05
    
    # Training settings
    fp16: bool = True
    gradient_checkpointing: bool = False  # Disabled due to compatibility issues
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    remove_unused_columns: bool = False
    logging_steps: int = 1
    
    # Data sampling (optimized for memory)
    n_samples: int = 5   # Reduced for memory efficiency (was 10)
    n_chunks: int = 5    # Reduced for memory efficiency (was 10)
    
    # Cross-validation
    k_folds: int = 5
    val_folds: int = 4


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Random seed
    random_seed: int = 13
    
    # Output paths
    output_prefix: str = ""
    
    # Cross-validation fold
    fold_id: int = 0
    
    def __post_init__(self):
        if not self.output_prefix:
            self.output_prefix = "output/"


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_fold_id_and_output(cls, fold_id: int, output_prefix: str, base_dir: str = None):
        """Create config with specific fold_id and output_prefix"""
        data_config = DataConfig()
        
        # Use provided base_dir or environment variable
        if base_dir is None:
            base_dir = data_config.dir
        else:
            data_config.dir = base_dir
        
        # Construct full paths by combining base_dir with relative paths
        import os.path
        data_config.full_df_name = os.path.join(base_dir, data_config.full_df_name)
        data_config.antibody_escape_f_name = os.path.join(base_dir, data_config.antibody_escape_f_name)
        data_config.dms_fasta_f_name = os.path.join(base_dir, data_config.dms_fasta_f_name)
        
        model_config = ModelConfig()
        # Use HuggingFace model by default instead of local DA_model
        model_config.da_model_name = model_config.model_name
        
        experiment_config = ExperimentConfig()
        experiment_config.fold_id = fold_id
        experiment_config.output_prefix = os.path.join(base_dir, "output", output_prefix)
        
        return cls(
            data=data_config,
            model=model_config,
            training=TrainingConfig(),
            experiment=experiment_config
        )


def get_default_config() -> Config:
    """Get default configuration"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        experiment=ExperimentConfig()
    )
