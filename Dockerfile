# Use a lightweight official Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port uvicorn will run on
EXPOSE 8080

# Run the FastAPI app with uvicorn, using the PORT environment variable
# provided by Cloud Run. The ${PORT:-8080} syntax means: use the PORT
# variable if it exists, otherwise default to 8080 (for local testing).
CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]