import joblib
from sklearn.linear_model import LogisticRegression

def build_and_train(X_train, y_train, model_path: str = "logistic_model.pkl"):
   
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model
