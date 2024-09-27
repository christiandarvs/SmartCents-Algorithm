import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = pd.read_csv("final_dataset.csv")

X = dataset.drop(columns=["Recommended_Course"])
y = dataset["Recommended_Course"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = SVC()
model.fit(X_train, y_train)
joblib.dump(model, "smartcents_model.joblib")
