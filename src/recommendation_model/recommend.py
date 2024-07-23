# src/models/recommendation_model/recommend.py

import torch
import numpy as np
from model import RecommendationModel

def make_recommendations(query_features_path, combined_features_path, model_path):
    # Load data
    query_features = np.load(query_features_path)
    combined_features = np.load(combined_features_path)
    
    # Load model
    input_dim = query_features.shape[1]
    model = RecommendationModel(input_dim, hidden_dim=512)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert to PyTorch tensors
    query_features_tensor = torch.tensor(query_features, dtype=torch.float32)
    combined_features_tensor = torch.tensor(combined_features, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        query_outputs = model(query_features_tensor)
        combined_outputs = model(combined_features_tensor)
    
    # Compute similarities
    similarities = torch.matmul(query_outputs, combined_outputs.T)
    return similarities.numpy()

# Example usage
query_features = np.load('data/processed/query_features.npy')
combined_features = np.load('data/processed/combined_features.npy')
model_path = 'models/recommendation_model/recommendation_model.pth'
recommendations = make_recommendations(query_features, combined_features, model_path)
print(recommendations)
