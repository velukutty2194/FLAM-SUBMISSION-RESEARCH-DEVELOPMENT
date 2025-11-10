import matplotlib.pyplot as plt
import numpy as np

def visualize_sigmoid(model, X, y):
  
  
    plt.figure(figsize=(7,5))
    plt.scatter(X, y, color='purple', label='Data Points')
    x_values = np.linspace(min(X.values), max(X.values), 100)
    y_prob = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
    plt.plot(x_values, y_prob, color='orange', linewidth=2, label='Sigmoid Curve')
    plt.xlabel('Feature X')
    plt.ylabel('Predicted Probability')
    plt.title('Logistic Regression Visualization')
    plt.legend()
    plt.show()
