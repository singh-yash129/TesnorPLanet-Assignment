# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models
RUN mkdir -p models

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
# We use a startup script to generate data and train model if not present (optional convenience)
# For now, just run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]