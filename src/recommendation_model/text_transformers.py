
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

def get_text_embeddings(texts, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings.append(cls_embedding)
    
    return embeddings

def save_text_embeddings(csv_path, output_path):
    df = pd.read_csv(csv_path)
    texts = df['processed_content'].tolist()
    embeddings = get_text_embeddings(texts)
    
    # Convert to DataFrame and save
    embedding_df = pd.DataFrame(embeddings)
    embedding_df.to_csv(output_path, index=False)
    print(f"Text embeddings saved to {output_path}")

if __name__ == "__main__":
    save_text_embeddings('data/processed/processed_text_data.csv', 'data/processed/text_embeddings.csv')
