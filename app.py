from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np

app = Flask(__name__)
mlflow.set_tracking_uri('http://localhost:5000')
model = mlflow.sklearn.load_model(model_uri=f"models:/Model_1/latest")  # Replace <your-run-id> from MLflow UI

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Expects JSON like {"data": [5.1, 3.5, 1.4, 0.2]}
    pred = model.predict([data])[0]
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)