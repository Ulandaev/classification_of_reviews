import json

import hydra
from omegaconf import DictConfig
from src.training.model_setup import setup_model

# from src.training.model_setup import setup_peft_config
from src.training.trainer import ReviewTrainer
from src.utils.config import get_file_path, setup_config


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Setup configuration
    cfg = setup_config(cfg)

    # Initialize trainer
    trainer = ReviewTrainer(cfg)

    # Load and prepare data
    data_path = get_file_path(cfg, "augmented_train")
    train_dataset, val_dataset, text_column = trainer.load_and_prepare_data(data_path)

    # PEFT
    # peft_config = setup_peft_config(cfg)
    model, tokenizer = setup_model(cfg, len(trainer.id_to_label))

    training_args = trainer.create_training_arguments()

    # обуч
    trained_trainer = trainer.train(
        train_dataset,
        val_dataset,
        model,
        tokenizer,
        text_column,
        training_args,
        cfg.model.classification.peft_method,
    )

    # сохранение модели и маппинг
    trained_trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    with open(
        f"{training_args.output_dir}/id_to_label.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {str(k): v for k, v in trainer.id_to_label.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
