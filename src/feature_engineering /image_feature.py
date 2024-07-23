import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def download_image(img_url):
    try:
        response = requests.get(img_url, timeout=10)  # Added timeout to handle slow responses
        response.raise_for_status()  # Check for HTTP errors
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading image from {img_url}: {e}")
        return None

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess an image for VGG16.
    """
    if img is None:
        return None
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_urls, model):
    """
    Extract features from a list of image URLs using a pre-trained model.
    """
    features = []
    for img_url in image_urls:
        img = download_image(img_url)
        img_array = preprocess_image(img)
        if img_array is not None:
            feature_vector = model.predict(img_array)
            features.append(feature_vector.flatten())
        else:
            # Use the shape of the model output layer to create a placeholder
            placeholder_shape = (model.output_shape[1],) if model.output_shape is not None else (4096,)  # Default to 4096 if output_shape is None
            features.append(np.zeros(placeholder_shape))
    return np.array(features)

def main():

    image_csv_path = '/Users/awnishranjan/Developer/NLP/Content/data/processed/processed_metadata.csv'
    feature_csv_path = '/Users/awnishranjan/Developer/NLP/Content/data/processed/image_features.csv'
 
    image_data = pd.read_csv(image_csv_path)

    if 'photo_image_url' not in image_data.columns:
        raise ValueError("The 'photo_image_url' column is not found in the CSV file.")
    top_images = image_data.head(50)
 
    image_urls = top_images['photo_image_url'].astype(str).tolist()
    base_model = VGG16(weights='imagenet', include_top=False, pooling='flatten')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    features = extract_features(image_urls, model)

    df_features = pd.DataFrame(features)
    df_features.to_csv(feature_csv_path, index=False)
    print(f"Image features saved to {feature_csv_path}")

if __name__ == "__main__":
    main()
