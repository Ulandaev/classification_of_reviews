import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm

from ..utils.config import get_file_path

class DataLabeler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.categories = cfg.data.labeling.categories
        self.few_shot_examples = cfg.data.labeling.few_shot_examples
        self.setup_model()
    
    def setup_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.model.labeling.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.cfg.model.labeling.quantization.bnb_4bit_compute_dtype)
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.labeling.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.labeling.name,
            quantization_config=quant_config,
            device_map="auto"
        )
    
    def generate_prompt(self, review, few_shot=True):
        categories_str = ', '.join(self.categories)
        system_prompt = self.cfg.model.labeling.prompt_templates.system
        
        messages = [{"role": "system", "content": system_prompt}]

        if few_shot:
            for ex in self.few_shot_examples:
                user_content = self.cfg.model.labeling.prompt_templates.user_template.format(review=ex['review'])
                assistant_content = self.cfg.model.labeling.prompt_templates.assistant_template.format(category=ex['category'])
                
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})

        user_content = self.cfg.model.labeling.prompt_templates.user_template.format(review=review)
        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def label_review(self, review):
        prompt = self.generate_prompt(review)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.cfg.model.labeling.generation.max_new_tokens,
                temperature=self.cfg.model.labeling.generation.temperature
            )
        
        generated = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], 
                                        skip_special_tokens=True).strip()
        
        match = re.search(r"(Категория:\s*)?([\w\s]+)", generated)
        if match and match.group(2).strip() in self.categories:
            return match.group(2).strip()
        return "нет товара"
    
    def label_dataframe(self, df, text_column=None):
        if text_column is None:
            text_column = self.cfg.data.labeling.columns.text
            
        labels = []
        for review in tqdm(df[text_column], desc="Labeling reviews"):
            label = self.label_review(review)
            labels.append(label)
        
        df[self.cfg.data.labeling.columns.label] = labels
        return df
    
    def label_from_file(self):
        """Complete labeling pipeline from file"""
        train_path = get_file_path(self.cfg, 'train')
        train_df = pd.read_csv(train_path)
        
        labeled_df = self.label_dataframe(train_df)
        
        output_path = get_file_path(self.cfg, 'labeled_train')
        labeled_df.to_csv(output_path, index=False)
        
        return labeled_df