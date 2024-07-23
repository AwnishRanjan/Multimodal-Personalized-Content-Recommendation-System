
import os
import pandas as pd

def load_data(data_dir):
    print(f"Loading data from directory: {data_dir}")
    
    photos_path = os.path.join(data_dir, 'photos.tsv')
    keywords_path = os.path.join(data_dir, 'keywords.tsv')
    collections_path = os.path.join(data_dir, 'collections.tsv')
    conversions_path = os.path.join(data_dir, 'conversions.tsv')
    colors_path = os.path.join(data_dir, 'colors.tsv')
    
    print(f"Photos path: {photos_path}")
    print(f"Keywords path: {keywords_path}")
    print(f"Collections path: {collections_path}")
    print(f"Conversions path: {conversions_path}")
    print(f"Colors path: {colors_path}")
    
    photos = pd.read_csv(photos_path, sep='\t')
    keywords = pd.read_csv(keywords_path, sep='\t')
    collections = pd.read_csv(collections_path, sep='\t')
    conversions = pd.read_csv(conversions_path, sep='\t')
    colors = pd.read_csv(colors_path, sep='\t')
    
    return photos, keywords, collections, conversions, colors

# if __name__ == "__main__":
#     data_dir = 'data/raw'
#     photos, keywords, collections, conversions, colors = load_data(data_dir)
#     print(photos.head())
#     print(keywords.head())
#     print(collections.head())
#     print(conversions.head())
#     print(colors.head())
