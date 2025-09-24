import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import json
import torch

class ReviewPredictor:
    def __init__(self, trainer, tokenizer, id_to_label):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.id_to_label = id_to_label
    
    def predict(self, texts, batch_size=32):
        all_predictions = []
        inference_times = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.trainer.model.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.trainer.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
            
            end_time = time.time()
            batch_time = (end_time - start_time) / len(batch_texts)
            inference_times.extend([batch_time] * len(batch_texts))
        
        predicted_categories = [self.id_to_label[str(pred)] for pred in all_predictions]
        
        return predicted_categories, inference_times