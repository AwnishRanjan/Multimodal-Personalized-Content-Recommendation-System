# Multimodal Personalized Content Recommendation System

## Overview

This project implements a multimodal content recommendation system that leverages both text and image data. It integrates natural language processing (NLP) and deep learning techniques to provide personalized content recommendations based on user preferences.

### Prerequisites

- Python 3.10 or later
- Required Python packages (install via `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AwnishRanjan/multimodal-content-recommender.git
    cd multimodal-content-recommender
    ```

2. Set up a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1. Place your raw data files in the `data/raw/` directory.
2. Run the preprocessing scripts to prepare the data:
    ```bash
    python src/data_preprocessing/text_preprocessing.py
    python src/data_preprocessing/image_preprocessing.py
    ```

### Feature Engineering

1. Extract features from text and images:
    ```bash
    python src/feature_engineering/text_feature_extraction.py
    python src/feature_engineering/image_feature_extraction.py
    ```

### Model Training and Evaluation

1. Train models:
    ```bash
    python src/models/text_model/train.py
    python src/models/image_model/train.py
    python src/models/recommendation_model/train.py
    ```

2. Evaluate models:
    ```bash
    python src/models/text_model/evaluate.py
    python src/models/image_model/evaluate.py
    python src/models/recommendation_model/evaluate.py
    ```

### Running the Recommendation Engine

1. Run the recommendation engine:
    ```bash
    python src/recommendation/recommendation_engine.py
    ```

## Contributing

If you'd like to contribute to this project, please fork the repository and help me to make it live with you web development and submit a pull request with your changes.




