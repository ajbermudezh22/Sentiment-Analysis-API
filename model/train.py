import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from datasets import load_dataset

def train_and_save_model():
    """
    Trains a sentiment analysis model and saves it.
    """
    print("Loading IMDb dataset...")
    # Load the dataset from Hugging Face
    imdb_dataset = load_dataset("imdb")

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(imdb_dataset['train'])
    test_df = pd.DataFrame(imdb_dataset['test'])
    
    # Combine for a larger dataset and then split
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Map labels to human-readable format for clarity, though model uses 0/1
    df['sentiment'] = df['label'].apply(lambda x: 'positive' if x == 1 else 'negative')
    
    print("Dataset loaded and preprocessed.")
    print(df.head())

    # Split the data
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data size: {len(X_train)}")
    print(f"Test data size: {len(X_test)}")

    # Create a scikit-learn pipeline
    # This chains the text vectorizer and the classifier together.
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])

    print("Training the model...")
    # Train the model
    sentiment_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = sentiment_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save the entire pipeline
    model_filepath = "model/sentiment_pipeline.joblib"
    joblib.dump(sentiment_pipeline, model_filepath)
    print(f"Model pipeline saved to {model_filepath}")

if __name__ == '__main__':
    train_and_save_model()