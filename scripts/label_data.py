import hydra
from omegaconf import DictConfig

from src.data.labeling import DataLabeler
from src.utils.config import setup_config

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Setup configuration
    cfg = setup_config(cfg)
    
    # Initialize labeler
    labeler = DataLabeler(cfg)
    
    # Label data
    labeled_df = labeler.label_from_file()
    print("Data labeling completed!")

if __name__ == "__main__":
    main()