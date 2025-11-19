import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# This dictionary will hold our model
# We load it on startup to avoid loading it for every request
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model during startup
    print("Loading sentiment analysis model...")
    # Adjust the path to go up one directory level
    model_path = "model/sentiment_pipeline.joblib"
    model_cache["sentiment_model"] = joblib.load(model_path)
    print("Model loaded successfully.")
    yield
    # Clean up the model during shutdown
    print("Clearing model cache.")
    model_cache.clear()

# Create the FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# Pydantic models for request and response
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    probability: float

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    """
    Predicts the sentiment of a given text.
    - Input: A JSON with a "text" field.
    - Output: A JSON with "sentiment" and "probability".
    """
    # Get the model from the cache
    pipeline = model_cache['sentiment_model']
    
    # Predict the sentiment (0 or 1)
    prediction = pipeline.predict([request.text])[0]
    
    # Predict the probability for the positive class
    # pipeline.predict_proba returns probabilities for [negative, positive]
    probability = pipeline.predict_proba([request.text])[0][1]
    
    # Map prediction to a human-readable label
    sentiment = "positive" if prediction == 1 else "negative"
    
    return PredictionResponse(sentiment=sentiment, probability=probability)