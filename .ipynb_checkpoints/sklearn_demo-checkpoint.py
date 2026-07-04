import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Loading Iris dataset...")
    # Load the classic Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {iris.target_names}")
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Initialize a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("\nTraining the Random Forest model...")
    clf.fit(X_train, y_train)
    
    print("Making predictions on the test set...")
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()
