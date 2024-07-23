import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def feature_engineering(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    X_array = X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)
    return X_scaled, vectorizer

def main():
    csv_file_path = 'data/processed/processed_text_data.csv'
    data = pd.read_csv(csv_file_path)
    print(f"Columns in the dataset: {data.columns}")
    if 'processed_content' not in data.columns:
        raise ValueError("The 'processed_content' column is not found in the CSV file.")

    texts = data['processed_content'].astype(str).tolist()
    X_scaled, vectorizer = feature_engineering(texts)
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame(X_scaled, columns=feature_names)
    feature_csv_path = 'data/processed/feature_data.csv'
    df.to_csv(feature_csv_path, index=False)
    print(f"Feature data saved to {feature_csv_path}")

if __name__ == "__main__":
    main()
