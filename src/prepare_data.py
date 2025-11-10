import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_dataset(file_path: str, test_ratio: float = 0.25, random_state: int = 101):
  
    data = pd.read_csv(file_path)
    X = data[['x']]
    y = data['y']
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)
