from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

def setup_peft_config(method="lora", lora_config=None):
    if method == "lora":
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            **lora_config
        )
    return None

def setup_model(model_name, peft_config, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    
    if peft_config:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer