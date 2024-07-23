# src/feature_engineering/image_feature_extraction.py

from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import pandas as pd

class ImageEmbedder:
    def __init__(self, model_name='resnet50'):
        self.model = getattr(models, model_name)(pretrained=True)
        self.model.eval()  # Set to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze().numpy()

def extract_image_features(csv_path):
    df = pd.read_csv(csv_path)
    image_embedder = ImageEmbedder()
    df['image_features'] = df['photo_image_url'].apply(lambda url: image_embedder.encode(url))
    features = np.vstack(df['image_features'].values)
    return features

# Example usage
image_features = extract_image_features('data/processed/processed_metadata.csv')
