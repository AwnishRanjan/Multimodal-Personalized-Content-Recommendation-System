import pandas as pd
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"""  
 load spacy and then TOkenisation and then lemmatiziation 

"""
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Spacy model not found")

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into string
    return ' '.join(tokens)

def preprocess_nlp_data(input_path, output_path):
    """Load, process, and save NLP text data."""
    # Load data
    data = pd.read_csv(input_path)

    # Debugging: Print the columns of the DataFrame
    print("Columns in the CSV file:", data.columns)

    # Check if 'content' column exists
    if 'content' not in data.columns:
        raise KeyError("The 'content' column is not found in the input file.")

    # Process text data
    data['processed_content'] = data['content'].astype(str).apply(preprocess_text)

    # Save processed data
    data.to_csv(output_path, index=False)
    print("Processed data saved to:", output_path)
    
input_path = 'data/raw/text_data.csv'
output_path = 'data/processed/processed_text_data.csv'


preprocess_nlp_data(input_path, output_path)
