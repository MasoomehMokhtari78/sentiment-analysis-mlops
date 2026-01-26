# Sentiment Analysis API

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready REST API for sentiment analysis that classifies text as positive, negative, or neutral. Built with FastAPI, scikit-learn, and transformers, following SOLID principles and industry best practices.

## Features

- **Two Model Approaches**:
  - **Baseline Model**: TF-IDF + Logistic Regression (fast, lightweight)
  - **Advanced Model**: DistilBERT (state-of-the-art accuracy)
  
- **Production-Ready API**:
  - REST endpoints with automatic documentation
  - Request validation using Pydantic
  - API key authentication
  - Rate limiting
  - CORS support
  - Security headers
  
- **MLOps Best Practices**:
  - MLflow experiment tracking
  - Model versioning
  - Comprehensive testing (>85% coverage)
  - Docker containerization
  - Health check endpoints

## Project Structure

```
sentiment-analysis-api/
├── src/
│   ├── data/               # Data processing modules
│   │   ├── processor.py    # Data loading and cleaning
│   │   └── __init__.py
│   ├── models/             # Model implementations
│   │   ├── base.py         # Base classes and interfaces
│   │   ├── baseline.py     # TF-IDF + Logistic Regression
│   │   ├── distilbert.py   # DistilBERT model
│   │   └── __init__.py
│   ├── api/                # API layer
│   │   ├── schemas.py      # Pydantic models
│   │   ├── security.py     # Authentication & security
│   │   ├── service.py      # Business logic
│   │   └── __init__.py
│   └── __init__.py
├── tests/                  # Unit tests
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── data/
│   ├── raw/                # Raw datasets
│   └── processed/          # Processed datasets
├── models/                 # Trained models
├── config.py               # Configuration management
├── main.py                 # FastAPI application
├── train.py                # Training script
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) Docker and Docker Compose

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd sentiment-analysis-api
```

2. **Create a virtual environment**:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Unix/MacOS
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and set your values
# Especially change the API_KEY for production!
```

5. **Prepare your dataset**:
```bash
# Place the Yelp review dataset in the parent directory
# Or update DATA_PATH in .env to point to your dataset
```

## Training Models

### Train Baseline Model (Recommended for Quick Start)

```bash
python train.py --model-type baseline --sample-size 50000
```

**Expected output**:
- Training time: ~2-5 minutes
- Accuracy: ~85-88%
- Model saved to: `models/baseline_model.pkl`

### Train DistilBERT Model

```bash
python train.py --model-type distilbert --sample-size 50000
```

**Expected output**:
- Training time: ~15-30 minutes (CPU) / ~5-10 minutes (GPU)
- Accuracy: ~90-93%
- Model saved to: `models/distilbert_model/`

### Train Both Models

```bash
python train.py --model-type both --sample-size 100000
```

### Training Options

```bash
python train.py --help

Options:
  --model-type {baseline,distilbert,both}  Type of model to train
  --data-path PATH                         Path to dataset
  --sample-size INT                        Number of samples to use
  --output-dir PATH                        Directory to save models
```

## Running the API

### Local Development

```bash
# Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Using Docker

**Quick Start (Recommended)**:

```bash
# Make sure you have trained the baseline model first
# Train if not already done:
python train.py --model-type baseline --sample-size 10000

# Start API with simple docker-compose (no MLflow)
docker-compose -f docker-compose.simple.yml up -d

# Access API at http://localhost:8000/docs
```

**With MLflow (Optional)**:

```bash
# Start both API and MLflow tracking server
docker-compose --profile mlflow up -d

# Access:
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

**Docker Commands**:
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Note**: See [DOCKER_MLFLOW_GUIDE.md](DOCKER_MLFLOW_GUIDE.md) for detailed Docker and MLflow setup.

### Using Docker (standalone)

```bash
# Build image
docker build -t sentiment-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_KEY=your-secure-key \
  -v $(pwd)/models:/app/models:ro \
  --name sentiment-api \
  sentiment-api
```

## API Usage

### Authentication

All prediction endpoints require an API key in the header:

```bash
X-API-Key: your-api-key-here
```

### Endpoints

#### 1. Health Check

```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "baseline",
  "version": "1.0.0",
  "timestamp": "2024-01-25T10:30:00Z"
}
```

#### 2. Single Prediction

```bash
POST /predict
```

**Request**:
```json
{
  "text": "This restaurant has amazing food and great service!"
}
```

**Response**:
```json
{
  "text": "This restaurant has amazing food and great service!",
  "sentiment": "positive",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.03,
    "positive": 0.95
  },
  "timestamp": "2024-01-25T10:30:00Z"
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

#### 3. Batch Prediction

```bash
POST /predict/batch
```

**Request**:
```json
{
  "texts": [
    "The food was excellent!",
    "Terrible experience, would not recommend.",
    "It was okay, nothing special."
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "text": "The food was excellent!",
      "sentiment": "positive",
      "confidence": 0.92,
      "probabilities": {
        "negative": 0.03,
        "neutral": 0.05,
        "positive": 0.92
      },
      "timestamp": "2024-01-25T10:30:00Z"
    },
    ...
  ],
  "total_count": 3,
  "timestamp": "2024-01-25T10:30:00Z"
}
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/predict/batch"
headers = {"X-API-Key": "your-api-key"}
data = {
    "texts": [
        "Great product!",
        "Not satisfied with the purchase."
    ]
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with verbose output
pytest -v
```

## MLflow Tracking

View experiment results and model metrics:

```bash
# Start MLflow UI (if not using docker-compose)
mlflow ui --port 5000

# Access at http://localhost:5000
```

MLflow tracks:
- Model parameters
- Training metrics (accuracy, F1-score, etc.)
- Confusion matrices
- Model artifacts

## Model Architecture

### Baseline Model
- **Vectorization**: TF-IDF (max 10,000 features, bigrams)
- **Classifier**: Logistic Regression (multinomial)
- **Advantages**: Fast training (~4s), low memory, interpretable
- **Performance**: 78.1% accuracy, 79.4% F1-score (on 10K Yelp reviews)

### DistilBERT Model
- **Base Model**: `distilbert-base-uncased`
- **Fine-tuning**: 3 epochs with linear warmup
- **Advantages**: State-of-the-art accuracy, context-aware
- **Performance**: 85.0% accuracy, 84.5% F1-score (on 10K Yelp reviews)

## Security Features

- API Key authentication
- Rate limiting (100 requests/minute default)
- CORS configuration
- Security headers (XSS, CSRF protection)
- Input validation and sanitization
- Non-root Docker user
- Request/response logging
- Error handling without information leakage

## Configuration

Edit `.env` file to configure:

```bash
# API Configuration
API_KEY=your-secure-api-key-here
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production

# Model Configuration
MODEL_TYPE=distilbert  # or baseline
MODEL_PATH=models/best_model.pkl
MAX_LENGTH=128
BATCH_SIZE=32

# Data Configuration
DATA_PATH=../yelp_academic_dataset_review.json
SAMPLE_SIZE=100000

# Security
RATE_LIMIT=100/minute
```

## Dataset Information

This project uses the **Yelp Academic Dataset**:
- **Source**: [Yelp Dataset](https://www.yelp.com/dataset)
- **Size**: 6.9M+ reviews, 150K+ businesses
- **Features**: Review text, star ratings (1-5)
- **Sentiment Mapping**:
  - 1-2 stars → Negative
  - 3 stars → Neutral
  - 4-5 stars → Positive

## Design Decisions

### SOLID Principles Implementation

1. **Single Responsibility**: Each class has one reason to change
   - `DataProcessor`: Data handling only
   - `ModelService`: Model operations only
   - `Security`: Authentication only

2. **Open/Closed**: Extensible without modification
   - Abstract base classes for models and data loaders
   - Easy to add new model types

3. **Liskov Substitution**: Models are interchangeable
   - All models implement `ModelInterface`
   - Can switch between baseline/DistilBERT seamlessly

4. **Interface Segregation**: Focused interfaces
   - Separate interfaces for data loading and cleaning
   - Minimal, specific API contracts

5. **Dependency Inversion**: Depend on abstractions
   - Dependency injection throughout
   - Configuration-driven design

### Technology Choices

- **FastAPI**: Modern, fast, automatic documentation
- **Pydantic**: Data validation and settings management
- **DistilBERT**: 40% smaller than BERT, 60% faster, 95% performance
- **MLflow**: Industry-standard experiment tracking
- **Docker**: Consistent deployment across environments

## Troubleshooting

### Model not loading
```bash
# Check model file exists
ls -la models/

# Verify MODEL_PATH in .env
cat .env | grep MODEL_PATH

# Check logs
docker-compose logs api
```

### Memory issues during training
```bash
# Reduce sample size
python train.py --model-type baseline --sample-size 20000

# Or use baseline model instead of DistilBERT
```

### API returns 503
```bash
# Model not loaded - train a model first
python train.py --model-type baseline --sample-size 50000

# Update MODEL_PATH in .env to point to trained model
```

## Performance Benchmarks

### Baseline Model (TF-IDF + Logistic Regression)
Trained on 10,000 Yelp reviews:

| Metric | Score |
|--------|-------|
| **Accuracy** | 78.10% |
| **Precision** | 81.19% |
| **Recall** | 78.10% |
| **F1-Score** | 79.39% |
| **Training Time** | ~4 seconds |

**Per-Class Performance:**
```
              precision    recall  f1-score   support
    negative       0.72      0.77      0.74       368
     neutral       0.31      0.43      0.36       228
    positive       0.92      0.84      0.88      1404
```

### DistilBERT Model
Trained on 10,000 Yelp reviews:

| Metric | Score |
|--------|-------|
| **Accuracy** | 85.05% |
| **Precision** | 84.06% |
| **Recall** | 85.05% |
| **F1-Score** | 84.47% |
| **Training Time** | ~15-20 minutes (CPU) |

**Per-Class Performance:**
```
              precision    recall  f1-score   support
    negative       0.83      0.83      0.83       368
     neutral       0.35      0.35      0.35       228
    positive       0.94      0.94      0.94      1404
```

**Model Comparison:**
- **Baseline**: Fast training (~4s), lightweight, good for production with limited resources (78% accuracy)
- **DistilBERT**: Higher accuracy (+7%), better context understanding, requires more compute (85% accuracy)

## CI/CD (Bonus)

For GitHub Actions integration, add `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src
```

## License

This project is licensed under the MIT License.

## Author

Developed for Senior AI/ML Engineer recruitment exercise.

## Acknowledgments

- Yelp for providing the academic dataset
- Hugging Face for transformer models
- FastAPI team for excellent framework

