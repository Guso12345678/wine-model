import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='Guso12345678', repo_name='wine-model', mlflow=True)

wine = datasets.load_wine()
X = wine.data
y = wine.target

with mlflow.start_run():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Las SVM son sensibles a la escala de los datos, así que normalizamos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernel = "linear"
    C = 1.0

    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, 'model.pkl')
    mlflow.sklearn.log_model(model, "wine-svm")
    mlflow.log_param("kernel", kernel)
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Precisión: {accuracy:.4f}")