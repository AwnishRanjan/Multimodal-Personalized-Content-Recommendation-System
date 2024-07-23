
import numpy as np 
from feature_engineering.text_feature_extraction import extract_text_features
from feature_engineering.image_feature_extraction import extract_image_features
from data_preparation.combine_features import combine_features
from models.recommendation_model.train import train_model
from models.recommendation_model.recommend import make_recommendations

def main():
    text_features = extract_text_features('data/processed/processed_text_data.csv')
    image_features = extract_image_features('data/processed/processed_metadata.csv')
    
    np.save('data/processed/text_features.npy', text_features)
    np.save('data/processed/image_features.npy', image_features)
  
    combine_features('data/processed/text_features.npy', 'data/processed/image_features.npy', 'data/processed/combined_features.npy')
    
    train_model('data/processed/combined_features.npy', 'models/recommendation_model/recommendation_model.pth')
    
    query_features = np.load('data/processed/query_features.npy') 
    combined_features = np.load('data/processed/combined_features.npy')
    recommendations = make_recommendations(query_features, combined_features, 'models/recommendation_model/recommendation_model.pth')

    print(recommendations)

if __name__ == "__main__":
    main()
