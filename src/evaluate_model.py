from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def assess_model(model, X_test, y_test):
   
   
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification_Report": classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics
