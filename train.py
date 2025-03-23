import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Train model
with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log params and metrics
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", accuracy)

    # Log model with explicit artifact location
    input_example = X_train[0].reshape(1, -1)  # Single iris sample: (1, 4)
    mlflow.sklearn.log_model(model, artifact_path="mlartifacts/model", input_example=input_example)
    
    print(f"Accuracy: {accuracy}")
