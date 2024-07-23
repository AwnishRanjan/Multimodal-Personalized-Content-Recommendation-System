# src/models/recommendation_model/train.py

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from model import RecommendationModel
import torch.nn as nn

def train_model(combined_features_path, output_model_path, epochs=10, learning_rate=0.001):
    # Load data
    combined_features = np.load(combined_features_path)
    X_train, X_val = train_test_split(combined_features, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    # Define model, loss function, and optimizer
    input_dim = X_train_tensor.shape[1]
    model = RecommendationModel(input_dim, hidden_dim=512)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    
    # Save the model
    torch.save(model.state_dict(), output_model_path)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = mean_squared_error(X_val_tensor.numpy(), val_outputs.numpy())
        print(f'Validation Loss: {val_loss}')

# Example usage
train_model('data/processed/combined_features.npy', 'models/recommendation_model/recommendation_model.pth')
