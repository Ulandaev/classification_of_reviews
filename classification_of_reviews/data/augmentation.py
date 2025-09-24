from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import pandas as pd

class DataAugmenter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.setup_model()
    
    def setup_model(self):
        quant_config = BitsAndBytesConfig(load_in_4bit=True, 
                                        bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=quant_config, 
            device_map="auto"
        )
    
    def paraphrase_review(self, review):
        system_prompt = "Ты помощник по перефразировке. Перефразируй отзыв, сохраняя исходный смысл."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Перефразируй следующий отзыв: {review}"}
        ]
        
        para_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, 
                                                       add_generation_prompt=True)
        inputs = self.tokenizer(para_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        
        return self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], 
                                   skip_special_tokens=True).strip()
    
    def augment_dataframe(self, df, text_column='text', label_column='label'):
        augmented_reviews = []
        augmented_labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting data"):
            para = self.paraphrase_review(row[text_column])
            augmented_reviews.append(para)
            augmented_labels.append(row[label_column])
        
        # Combine original and augmented data
        aug_df = pd.DataFrame({
            text_column: augmented_reviews + df[text_column].tolist(),
            label_column: augmented_labels + df[label_column].tolist()
        })
        
        return aug_df