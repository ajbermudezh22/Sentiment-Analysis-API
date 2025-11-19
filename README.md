# 🚀 Real-Time Sentiment Analysis API

> **🎯 Live Demo:** [Try the interactive web application here!](https://sentiment-analysis-api-25rx6onjnfoqgqd6kpy8qm.streamlit.app/)

A production-ready, end-to-end machine learning project that demonstrates the complete ML lifecycle: from training a classic scikit-learn model to deploying it as a scalable REST API with an interactive web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Key Learnings](#key-learnings)

---

## 🎯 Overview

This project showcases a **classic machine learning** workflow (as opposed to modern LLM-based approaches) using traditional data science tools. It demonstrates:

- **Data Science:** Loading, preprocessing, and training a sentiment analysis model using pandas and scikit-learn
- **ML Engineering:** Serializing and serving a trained model via a REST API
- **DevOps:** Containerizing the application with Docker
- **Cloud Deployment:** Deploying to Google Cloud Run (containerized) and Streamlit Community Cloud (frontend)
- **Full-Stack Development:** Building a decoupled architecture with a Python backend API and a Python frontend UI

The model achieves **~89% accuracy** on the IMDb movie review dataset, classifying text as either positive or negative sentiment.

---

## 🏗️ Architecture

This project follows a **microservices architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                     │
│  https://sentiment-analysis-api-25rx6onjnfoqgqd6kpy8qm...   │
│  - User-friendly web interface                              │
│  - Real-time sentiment analysis                             │
│  - Visual feedback and confidence scores                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST Request
                       │ (JSON: {"text": "..."})
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend API (FastAPI + scikit-learn)            │
│  https://sentiment-api-service-105420428646.us-central1...  │
│  - Loads trained model on startup                           │
│  - Processes text through TF-IDF vectorization              │
│  - Returns sentiment prediction + confidence score          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Model (scikit-learn)                   │
│  - TF-IDF Vectorizer (text → numerical features)           │
│  - Logistic Regression Classifier                           │
│  - Trained on 50,000 IMDb movie reviews                    │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Decoupled Frontend/Backend:** The UI and API are completely independent, allowing for easy scaling and updates
- **Model Caching:** The model is loaded once on API startup, not on every request (efficient!)
- **Production-Ready Pipeline:** Using scikit-learn's `Pipeline` ensures consistent preprocessing between training and inference

---

## ✨ Features

- ✅ **Real-Time Predictions:** Get instant sentiment analysis results
- ✅ **Confidence Scores:** See how confident the model is in its predictions
- ✅ **Interactive UI:** Beautiful, user-friendly web interface built with Streamlit
- ✅ **RESTful API:** Well-documented API endpoint for programmatic access
- ✅ **Scalable Deployment:** Containerized backend that auto-scales on Google Cloud Run
- ✅ **High Accuracy:** ~89% accuracy on test data

---

## 🛠️ Tech Stack

### Data Science & Machine Learning
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning pipeline (TF-IDF + Logistic Regression)
- **joblib** - Model serialization
- **datasets (Hugging Face)** - Dataset loading

### Backend API
- **FastAPI** - Modern, fast web framework for building APIs
- **Pydantic** - Data validation using Python type annotations
- **uvicorn** - ASGI server for running FastAPI

### Frontend
- **Streamlit** - Rapid web app development in pure Python

### DevOps & Deployment
- **Docker** - Containerization
- **Google Cloud Run** - Serverless container platform (backend)
- **Streamlit Community Cloud** - Free hosting for Streamlit apps (frontend)

---

## 📁 Project Structure

```
Real-Time Sentiment Analysis API/
├── api/
│   └── index.py                 # FastAPI application (backend)
├── model/
│   ├── train.py                 # Model training script
│   └── sentiment_pipeline.joblib # Trained model artifact
├── sentiment-ui-streamlit/
│   ├── frontend_app.py          # Streamlit UI (frontend)
│   └── requirements.txt         # Frontend dependencies
├── tests/
│   └── test_api.py             # API tests
├── Dockerfile                   # Container configuration
├── requirements.txt             # Backend dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerization)
- Google Cloud account (for deployment)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Real-Time Sentiment Analysis API"
   ```

2. **Train the model:**
   ```bash
   python model/train.py
   ```
   This will download the IMDb dataset, train the model, and save it as `model/sentiment_pipeline.joblib`.

3. **Run the API locally:**
   ```bash
   pip install -r requirements.txt
   uvicorn api.index:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`

4. **Run the frontend locally:**
   ```bash
   cd sentiment-ui-streamlit
   pip install -r requirements.txt
   streamlit run frontend_app.py
   ```
   The UI will open in your browser automatically.

### Testing the API

You can test the API using `curl` or PowerShell's `Invoke-RestMethod`:

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"text": "This movie was absolutely fantastic!"}'
```

**curl (Linux/Mac):**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

**Expected Response:**
```json
{
  "sentiment": "positive",
  "probability": 0.987654321
}
```

---

## 📚 API Documentation

### Base URL

**Production:** `https://sentiment-api-service-105420428646.us-central1.run.app`

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Sentiment Analysis API is running!"
}
```

#### `POST /predict`
Predicts the sentiment of a given text.

**Request Body:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "probability": 0.987654321
}
```

**Response Fields:**
- `sentiment` (string): Either `"positive"` or `"negative"`
- `probability` (float): Confidence score for the predicted class (0.0 to 1.0)

**Interactive API Docs:** Visit `https://sentiment-api-service-105420428646.us-central1.run.app/docs` for automatic interactive documentation powered by FastAPI.

---

## ☁️ Deployment

### Backend Deployment (Google Cloud Run)

1. **Build the Docker image:**
   ```bash
   docker build -t sentiment-api .
   ```

2. **Tag and push to Google Artifact Registry:**
   ```bash
   docker tag sentiment-api us-central1-docker.pkg.dev/YOUR_PROJECT_ID/sentiment-repo/sentiment-api:latest
   docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/sentiment-repo/sentiment-api:latest
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy sentiment-api-service \
     --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/sentiment-repo/sentiment-api:latest \
     --region us-central1 \
     --allow-unauthenticated \
     --memory=1Gi
   ```

**Key Configuration:**
- **Memory:** 1 GiB (required for loading the scikit-learn model)
- **Port:** Automatically configured via `PORT` environment variable
- **Scaling:** Auto-scales from 0 to multiple instances based on traffic

### Frontend Deployment (Streamlit Community Cloud)

1. Push your `sentiment-ui-streamlit` folder to a GitHub repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy! Streamlit automatically detects and deploys your app

---

## 🎓 Key Learnings

This project demonstrates several important production ML concepts:

1. **Model Serialization:** Using `joblib` to save and load trained scikit-learn pipelines
2. **API Design:** Building RESTful APIs with proper request/response models using Pydantic
3. **Containerization:** Creating portable Docker images that work consistently across environments
4. **Cloud Deployment:** Understanding the differences between container platforms (Cloud Run) and serverless functions (Vercel/Lambda)
5. **Microservices Architecture:** Decoupling frontend and backend for better scalability
6. **Production Considerations:** Memory limits, CORS configuration, model caching strategies
7. **Classic ML Stack:** Demonstrating proficiency with pandas, scikit-learn, and traditional ML workflows (as opposed to LLM-based approaches)

---

## 📊 Model Performance

- **Training Dataset:** 40,000 IMDb movie reviews
- **Test Dataset:** 10,000 IMDb movie reviews
- **Accuracy:** ~89.27%
- **Model Type:** TF-IDF Vectorizer + Logistic Regression
- **Features:** Unigrams and bigrams, English stop words removed

---

## 🔗 Links

- **Live Demo (Frontend):** [https://sentiment-analysis-api-25rx6onjnfoqgqd6kpy8qm.streamlit.app/](https://sentiment-analysis-api-25rx6onjnfoqgqd6kpy8qm.streamlit.app/)
- **API Endpoint:** [https://sentiment-api-service-105420428646.us-central1.run.app](https://sentiment-api-service-105420428646.us-central1.run.app)
- **Interactive API Docs:** [https://sentiment-api-service-105420428646.us-central1.run.app/docs](https://sentiment-api-service-105420428646.us-central1.run.app/docs)

---

## 📝 License

This project is open source and available for educational purposes.

---

## 👤 Author

Built as a portfolio project to demonstrate end-to-end ML engineering skills, from data science to production deployment.

---

**⭐ If you found this project helpful or interesting, please consider giving it a star!**