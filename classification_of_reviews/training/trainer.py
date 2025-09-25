import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments

# from ..utils.config import get_file_path
from ..utils.helpers import set_seed

# from .metrics import compute_metrics


class ReviewTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg.seed)

    def load_and_prepare_data(self, data_path):
        df = pd.read_csv(data_path)
        text_column = self.cfg.data.labeling.columns.text
        label_column = self.cfg.data.labeling.columns.label

        # фильрация валидных категорий
        df = df[df[label_column].isin(self.cfg.data.labeling.categories)]
        unique_categories = sorted(df[label_column].unique())

        # маппинг
        self.label_to_id = {label: i for i, label in enumerate(unique_categories)}
        self.id_to_label = {i: label for i, label in enumerate(unique_categories)}

        df["labels"] = df[label_column].map(self.label_to_id)

        # Split data
        train_df, val_df = train_test_split(
            df,
            test_size=self.cfg.data.preprocessing.train_test_split.test_size,
            random_state=self.cfg.data.preprocessing.train_test_split.random_state,
            shuffle=self.cfg.data.preprocessing.train_test_split.shuffle,
        )

        return (
            Dataset.from_pandas(train_df[[text_column, "labels"]]),
            Dataset.from_pandas(val_df[[text_column, "labels"]]),
            text_column,
        )

    def tokenize_data(self, tokenizer, datasets, text_column):
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                padding=self.cfg.data.preprocessing.tokenization.padding,
                truncation=self.cfg.data.preprocessing.tokenization.truncation,
                max_length=self.cfg.data.preprocessing.tokenization.max_length,
                return_tensors=self.cfg.data.preprocessing.tokenization.return_tensors,
            )

        tokenized_datasets = {}
        for name, dataset in datasets.items():
            tokenized_datasets[name] = dataset.map(
                tokenize_function, batched=True, remove_columns=[text_column]
            )

        return tokenized_datasets

    def create_training_arguments(self):
        return TrainingArguments(
            output_dir=self.cfg.training.output_dir,
            overwrite_output_dir=self.cfg.training.overwrite_output_dir,
            num_train_epochs=self.cfg.training.num_train_epochs,
            per_device_train_batch_size=self.cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.training.per_device_eval_batch_size,
            learning_rate=self.cfg.training.optimizer.learning_rate,
            weight_decay=self.cfg.training.optimizer.weight_decay,
            warmup_ratio=self.cfg.training.optimizer.warmup_ratio,
            logging_dir=self.cfg.training.logging_dir,
            logging_steps=self.cfg.training.logging_steps,
            eval_strategy=self.cfg.training.eval_strategy,
            save_strategy=self.cfg.training.save_strategy,
            load_best_model_at_end=self.cfg.training.load_best_model_at_end,
            metric_for_best_model=self.cfg.training.metric_for_best_model,
            greater_is_better=self.cfg.training.greater_is_better,
            fp16=self.cfg.training.fp16,
            dataloader_pin_memory=self.cfg.training.dataloader_pin_memory,
            report_to=self.cfg.training.report_to,
            seed=self.cfg.seed,
        )
