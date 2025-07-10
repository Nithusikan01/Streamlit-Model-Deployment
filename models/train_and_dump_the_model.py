from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd

def train_and_dump_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'random_forest_classifier.joblib')
    print("Model trained and saved successfully!")

def save_testdata(X_test):
    df_test = pd.DataFrame(X_test)
    df_test.to_csv('test_data.csv', index=False)
    print("Test data saved successfully!")

if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_and_dump_model(X_train, y_train)
    save_testdata(X_test)
    print("Training and test data preparation completed.")