# linear_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic data for linear regression.
    Features (X) are randomly generated, and the target (y) is a linear combination of X
    plus some random noise.
    """
    print("Generating synthetic data...")
    np.random.seed(42) # for reproducibility

    # Generate features: two independent variables
    X = np.random.rand(num_samples, 2) * 10 # Features between 0 and 10

    # Generate target variable: y = 2*x1 + 3*x2 + 5 + noise
    true_coefficients = np.array([2, 3])
    intercept = 5
    noise = np.random.randn(num_samples) * 2 # Add some Gaussian noise

    y = X @ true_coefficients + intercept + noise

    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['target'] = y
    print(f"Generated {num_samples} samples with 2 features.")
    return df

def train_and_save_model(df):
    """
    Trains a linear regression model, evaluates it, and saves the trained model
    using pickle.
    """
    print("Starting model training and evaluation...")

    # Define features (X) and target (y)
    X = df[['feature_1', 'feature_2']]
    y = df['target']

    # Split data into training (60%), validation (20%), and testing (20%) sets
    # First, split into training (80%) and temp (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Then, split temp (80% of original, which is 80%*0.2 = 16% of original) into
    # training (60% of original) and validation (20% of original)
    # The new test_size for this split is 0.25 (0.20 / 0.80 = 0.25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.2/0.8), random_state=42 # 0.25 ensures 20% validation
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    print("Training Linear Regression model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate on the validation set
    print("\nEvaluating model on validation set...")
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"Validation Mean Squared Error (MSE): {val_mse:.4f}")
    print(f"Validation R-squared (R2): {val_r2:.4f}")

    # Evaluate on the test set (final evaluation)
    print("\nEvaluating model on test set...")
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Test Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"Test R-squared (R2): {test_r2:.4f}")

    # Save the trained model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True) # Create 'models' directory if it doesn't exist
    model_filename = os.path.join(model_dir, 'linear_regression_model.pkl')

    print(f"\nSaving model to {model_filename}...")
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully!")

    print("\nModel Coefficients:")
    for i, coef in enumerate(model.coef_):
        print(f"  Feature {i+1}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

if __name__ == "__main__":
    data_df = generate_synthetic_data()
    train_and_save_model(data_df)
