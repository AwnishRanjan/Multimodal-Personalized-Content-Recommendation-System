# src/data_preparation/combine_features.py

import numpy as np
import pandas as pd

def combine_features(text_features_path, image_features_path, output_path):
    text_features = np.load(text_features_path)
    image_features = np.load(image_features_path)
    combined_features = np.concatenate([text_features, image_features], axis=1)
    np.save(output_path, combined_features)
    return combined_features

# Example usage
combine_features('data/processed/text_features.npy', 'data/processed/image_features.npy', 'data/processed/combined_features.npy')
