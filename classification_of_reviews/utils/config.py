import os
import torch
from omegaconf import DictConfig, OmegaConf

def setup_config(cfg: DictConfig) -> DictConfig:
    """Setup configuration with runtime adjustments"""
    # Create a copy to avoid modifying the original
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Set device-specific settings
    cfg.training.fp16 = torch.cuda.is_available()
    
    # Set model dtype based on hardware
    if torch.cuda.is_available():
        cfg.model.classification.torch_dtype = "float16"
    else:
        cfg.model.classification.torch_dtype = "float32"
    
    # Create directories
    os.makedirs(cfg.paths.processed_data, exist_ok=True)
    os.makedirs(cfg.paths.models, exist_ok=True)
    os.makedirs(cfg.paths.results, exist_ok=True)
    os.makedirs(cfg.paths.logs, exist_ok=True)
    
    return cfg

def get_file_path(cfg: DictConfig, file_type: str) -> str:
    """Get full file path from configuration"""
    file_mapping = {
        'train': cfg.data.labeling.file_names.train,
        'test': cfg.data.labeling.file_names.test,
        'labeled_train': cfg.data.labeling.file_names.labeled_train,
        'augmented_train': cfg.data.labeling.file_names.augmented_train,
        'test_predictions': cfg.data.labeling.file_names.test_predictions
    }
    
    if file_type in ['train', 'test']:
        base_dir = cfg.paths.raw_data
    else:
        base_dir = cfg.paths.processed_data
    
    return os.path.join(base_dir, file_mapping[file_type])