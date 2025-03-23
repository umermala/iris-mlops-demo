
# Iris MLOps Demo

This project demonstrates a basic MLOps workflow: training a machine learning model, logging it with MLflow, and serving it via a Flask API. The model predicts iris flower species (Setosa, Versicolor, Virginica) using the classic Iris dataset.

## Project Structure

```
iris-mlops-demo/
├── app.py          # Flask API to serve predictions
├── train.py        # Trains and logs the model with MLflow
├── requirements.txt# Python dependencies
├── Dockerfile      # Containerizes the Flask app
├── .gitignore      # Excludes generated files
└── README.md       # This file
```

## Prerequisites

- Python 3.8+
- Git
- Docker (optional, for containerization)
- MLflow server (optional, for remote tracking)

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com//iris-mlops-demo.git
cd iris-mlops-demo
```

### 2. Create a Virtual Environment
```bash
python -m venv mlops_env
```

#### Windows
```bash
mlops_env\Scripts\activate.bat
```

#### Linux/Mac
```bash
source mlops_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

#### Start MLflow UI (optional, for visualization):
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser.

#### Run the Training Script:
```bash
python train.py
```
This will train a LogisticRegression model on the Iris dataset and log parameters, metrics, and the model to MLflow (default: `mlruns/` or `http://localhost:5000` if set).

Note the Run ID: Check the MLflow UI for the run ID (e.g., `a1b2c3d4e5f6g7h8i9j0`) under experiment "0".

### Serving Predictions

#### Update `app.py`:
Replace:
```python
model = mlflow.sklearn.load_model("runs://model")
```
with the run ID from training.

#### Run the Flask App:
```bash
python app.py
```
The API will run at `http://localhost:5000/predict`.

#### Test the API:

Use `curl`:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [5.1, 3.5, 1.4, 0.2]}' http://localhost:5000/predict
```

Or Postman: POST to `http://localhost:5000/predict` with JSON `{"data": [5.1, 3.5, 1.4, 0.2]}`.

**Response**:
```json
{"prediction": 0}
```
(0 = Setosa, 1 = Versicolor, 2 = Virginica).

### Docker Deployment (Optional)

#### Build the Docker Image:
```bash
docker build -t iris-mlops-demo .
```

#### Run the Container:
```bash
docker run -p 5000:5000 iris-mlops-demo
```

#### Test: Same as above.

## Notes

- If using a remote MLflow server, ensure `mlflow.set_tracking_uri()` in `train.py` matches your server’s address (e.g., `http://localhost:5000`).
- Model artifacts are stored in `mlartifacts/` locally—adjust paths if needed.
- Run `train.py` before `app.py` to generate a model for the API to load.
