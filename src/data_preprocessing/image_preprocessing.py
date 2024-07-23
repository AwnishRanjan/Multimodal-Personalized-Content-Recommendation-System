import os
import pandas as pd

def load_metadata(data_dir):
    photos_path = os.path.join(data_dir, 'photos.csv')
    keywords_path = os.path.join(data_dir, 'keywords.csv')
    collections_path = os.path.join(data_dir, 'collections.csv')
    conversions_path = os.path.join(data_dir, 'conversions.csv')
    colors_path = os.path.join(data_dir, 'colors.csv')

    photos_chunks = pd.read_csv(photos_path, chunksize=10000)
    keywords_chunks = pd.read_csv(keywords_path, chunksize=10000)
    collections_chunks = pd.read_csv(collections_path, chunksize=10000)
    conversions_chunks = pd.read_csv(conversions_path, chunksize=10000)
    colors_chunks = pd.read_csv(colors_path, chunksize=10000)

    return photos_chunks, keywords_chunks, collections_chunks, conversions_chunks, colors_chunks

def preprocess_metadata(data_dir):
    # Load the data in chunks
    photos_chunks, keywords_chunks, collections_chunks, conversions_chunks, colors_chunks = load_metadata(data_dir)

    # List to store processed data chunks
    processed_data_list = []

    for photos_chunk in photos_chunks:
        # Handle missing values in photos chunk
        photos_chunk.dropna(subset=['photo_id'], inplace=True)
        photos_chunk.fillna({
            'photo_description': '',
            'exif_camera_make': 'Unknown',
            'exif_camera_model': 'Unknown',
            'photo_location_name': 'Unknown',
            'photo_location_country': 'Unknown',
            'photo_location_city': 'Unknown',
            'stats_views': photos_chunk['stats_views'].mean(),
            'stats_downloads': photos_chunk['stats_downloads'].mean()
        }, inplace=True)

        for keywords_chunk in keywords_chunks:
            # Handle missing values in keywords chunk
            keywords_chunk.dropna(subset=['photo_id'], inplace=True)
            keywords_chunk.fillna({'ai_service_1_confidence': 0, 'ai_service_2_confidence': 0, 'suggested_by_user': 'f'}, inplace=True)

            for collections_chunk in collections_chunks:
                # Handle missing values in collections chunk
                collections_chunk.dropna(subset=['photo_id'], inplace=True)
                collections_chunk.fillna({'collection_title': 'Unknown'}, inplace=True)

                for conversions_chunk in conversions_chunks:
                    # Handle missing values in conversions chunk
                    conversions_chunk.dropna(subset=['photo_id'], inplace=True)
                    conversions_chunk.fillna({'conversion_country': 'Unknown'}, inplace=True)

                    for colors_chunk in colors_chunks:
                        # Handle missing values in colors chunk
                        colors_chunk.dropna(subset=['photo_id'], inplace=True)
                        colors_chunk.fillna({'keyword': 'Unknown', 'ai_coverage': 0, 'ai_score': 0}, inplace=True)

                        # Merge all chunks on photo_id
                        merged_chunk = photos_chunk.merge(keywords_chunk, on='photo_id', how='left')
                        merged_chunk = merged_chunk.merge(collections_chunk, on='photo_id', how='left')
                        merged_chunk = merged_chunk.merge(conversions_chunk, on='photo_id', how='left')
                        merged_chunk = merged_chunk.merge(colors_chunk, on='photo_id', how='left')

                        # Append the processed chunk to the list
                        processed_data_list.append(merged_chunk)

    # Concatenate all processed chunks
    processed_data = pd.concat(processed_data_list, ignore_index=True)

    # Save preprocessed metadata
    processed_metadata_path = os.path.join('data/processed', 'processed_metadata.csv')
    processed_data.to_csv(processed_metadata_path, index=False)
    print(f"Processed metadata saved to {processed_metadata_path}")

if __name__ == "__main__":
    data_dir = 'data/raw'
    preprocess_metadata(data_dir)
