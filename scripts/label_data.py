import hydra
from omegaconf import DictConfig

from classification_of_reviews.data.labeling import DataLabeler
from classification_of_reviews.utils.config import setup_config


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    # разметчик
    labeler = DataLabeler(cfg)

    # разметка данных
    labeled_df = labeler.label_from_file()
    labeled_df.to_csv("data/labeled_data.csv", index=False, encoding="utf-8")
    print("Разметка завершена")


if __name__ == "__main__":
    main()
