from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd

class TextEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average pooling

def extract_text_features(csv_path):
    df = pd.read_csv(csv_path)
    text_embedder = TextEmbedder()
    df['text_features'] = df['processed_content'].apply(lambda x: text_embedder.encode(x))
    features = np.vstack(df['text_features'].values)
    return features

# Example usage
text_features = extract_text_features('data/processed/processed_text_data.csv')
