"""
MLflow Pipeline Components
This file contains component definitions for the MLflow pipeline
Each component is a reusable building block for the ML pipeline
"""
from kfp import dsl
from typing import NamedTuple


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['dvc', 'pandas']
)
def data_extraction(
    data_path: str,
    dvc_remote: str,
    output_data: dsl.Output[dsl.Dataset]
):
    """
    Data Extraction Component
    Fetches versioned dataset from DVC remote storage
    
    Inputs:
        - data_path (str): Path to the data file in DVC (e.g., 'data/raw/raw_data.csv')
        - dvc_remote (str): DVC remote storage location
        
    Outputs:
        - output_data (Dataset): Extracted dataset artifact
    """
    import subprocess
    import pandas as pd
    import os
    
    print(f"Extracting data from DVC remote: {dvc_remote}")
    print(f"Data path: {data_path}")
    
    # Initialize DVC and configure remote
    subprocess.run(['dvc', 'init', '--no-scm'], check=False)
    subprocess.run(['dvc', 'remote', 'add', '-d', 'myremote', dvc_remote], check=False)
    
    # Pull data from DVC
    try:
        result = subprocess.run(['dvc', 'pull', data_path], 
                              capture_output=True, text=True, check=True)
        print(f"DVC pull output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"DVC pull failed: {e.stderr}")
        # If DVC pull fails, try to read the file directly
        print("Attempting to read file directly...")
    
    # Load the data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Save to output artifact
    df.to_csv(output_data.path, index=False)
    
    print(f"Data extraction completed. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'numpy']
)
def data_preprocessing(
    input_data: dsl.Input[dsl.Dataset],
    train_data: dsl.Output[dsl.Dataset],
    test_data: dsl.Output[dsl.Dataset],
    test_size: float = 0.2,
    random_state: int = 42
) -> NamedTuple('Outputs', [('train_samples', int), ('test_samples', int), ('n_features', int)]):
    """
    Data Preprocessing Component
    Handles cleaning, scaling, and splitting data into train/test sets
    
    Inputs:
        - input_data (Dataset): Raw input dataset
        - test_size (float): Proportion of dataset for testing (default: 0.2)
        - random_state (int): Random seed for reproducibility (default: 42)
        
    Outputs:
        - train_data (Dataset): Preprocessed training dataset
        - test_data (Dataset): Preprocessed testing dataset
        - train_samples (int): Number of training samples
        - test_samples (int): Number of test samples
        - n_features (int): Number of features
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    
    print("Starting data preprocessing...")
    
    # Load data
    df = pd.read_csv(input_data.path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Handle missing values
    print("Checking for missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        # Fill missing values with median for numerical columns
        df = df.fillna(df.median(numeric_only=True))
        print("Missing values filled with median")
    else:
        print("No missing values found")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Standardize features (scale to mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized using StandardScaler")
    
    # Create DataFrames with scaled data
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    
    # Save to output artifacts
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    print(f"Preprocessing completed successfully")
    print(f"Train data saved: {train_data.path}")
    print(f"Test data saved: {test_data.path}")
    
    # Return metrics
    outputs = namedtuple('Outputs', ['train_samples', 'test_samples', 'n_features'])
    return outputs(
        train_samples=int(X_train.shape[0]),
        test_samples=int(X_test.shape[0]),
        n_features=int(X.shape[1])
    )


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'numpy', 'joblib']
)
def model_training(
    train_data: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model],
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    random_state: int = 42
) -> NamedTuple('Outputs', [('train_score', float), ('feature_importance_top3', str)]):
    """
    Model Training Component
    Trains a Random Forest Classifier on the training data and saves the model artifact
    
    Inputs:
        - train_data (Dataset): Preprocessed training dataset
        - n_estimators (int): Number of trees in the forest (default: 100)
        - max_depth (int): Maximum depth of trees (default: 10)
        - min_samples_split (int): Minimum samples required to split node (default: 2)
        - random_state (int): Random seed for reproducibility (default: 42)
        
    Outputs:
        - model_output (Model): Trained Random Forest model artifact
        - train_score (float): Training R² score
        - feature_importance_top3 (str): Top 3 most important features
        
    Description:
        This component trains a Random Forest Regressor on housing price data.
        The model learns patterns from the training data and is saved as a pickle file.
        It also computes feature importance to understand which features contribute most.
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    import joblib
    from collections import namedtuple
    
    print("="*50)
    print("MODEL TRAINING COMPONENT")
    print("="*50)
    
    # Load training data
    train_df = pd.read_csv(train_data.path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    print(f"\nTraining data loaded:")
    print(f"  - Samples: {X_train.shape[0]}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Feature names: {list(X_train.columns)}")
    
    # Initialize Random Forest Regressor
    print(f"\nInitializing Random Forest Regressor with:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - min_samples_split: {min_samples_split}")
    print(f"  - random_state: {random_state}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    
    print(f"\nTraining Performance:")
    print(f"  - R² Score: {train_r2:.4f}")
    print(f"  - MSE: {train_mse:.4f}")
    print(f"  - RMSE: {train_rmse:.4f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance (Top 5):")
    print(feature_importance.head())
    
    # Get top 3 features as string
    top3_features = ', '.join([
        f"{row['feature']}({row['importance']:.3f})" 
        for _, row in feature_importance.head(3).iterrows()
    ])
    
    # Save model using joblib
    joblib.dump(model, model_output.path)
    print(f"\nModel saved to: {model_output.path}")
    
    print("="*50)
    
    # Return outputs
    outputs = namedtuple('Outputs', ['train_score', 'feature_importance_top3'])
    return outputs(train_score=float(train_r2), feature_importance_top3=top3_features)


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'numpy', 'joblib']
)
def model_evaluation(
    test_data: dsl.Input[dsl.Dataset],
    model: dsl.Input[dsl.Model],
    metrics_output: dsl.Output[dsl.Metrics]
) -> NamedTuple('Outputs', [('test_r2', float), ('test_rmse', float), 
                             ('test_mae', float), ('test_mape', float)]):
    """
    Model Evaluation Component
    Loads the trained model, evaluates it on test set, and saves metrics
    
    Inputs:
        - test_data (Dataset): Preprocessed test dataset
        - model (Model): Trained model artifact
        
    Outputs:
        - metrics_output (Metrics): Evaluation metrics file (JSON format)
        - test_r2 (float): Test R² score
        - test_rmse (float): Test Root Mean Squared Error
        - test_mae (float): Test Mean Absolute Error
        - test_mape (float): Test Mean Absolute Percentage Error
        
    Description:
        This component evaluates the trained model on unseen test data.
        It computes multiple regression metrics including R², RMSE, MAE, and MAPE.
        Metrics are saved to a JSON file for tracking and comparison.
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        r2_score, mean_squared_error, 
        mean_absolute_error, mean_absolute_percentage_error
    )
    import joblib
    import json
    from collections import namedtuple
    
    print("="*50)
    print("MODEL EVALUATION COMPONENT")
    print("="*50)
    
    # Load test data
    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"\nTest data loaded:")
    print(f"  - Samples: {X_test.shape[0]}")
    print(f"  - Features: {X_test.shape[1]}")
    
    # Load trained model
    trained_model = joblib.load(model.path)
    print(f"\nModel loaded from: {model.path}")
    print(f"Model type: {type(trained_model).__name__}")
    
    # Make predictions
    print("\nMaking predictions on test data...")
    y_pred = trained_model.predict(X_test)
    
    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
    
    # Calculate additional metrics
    max_error = np.max(np.abs(y_test - y_pred))
    median_error = np.median(np.abs(y_test - y_pred))
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"R² Score:                    {r2:.4f}")
    print(f"Root Mean Squared Error:     {rmse:.4f}")
    print(f"Mean Absolute Error:         {mae:.4f}")
    print(f"Mean Absolute % Error:       {mape:.2f}%")
    print(f"Maximum Error:               {max_error:.4f}")
    print(f"Median Absolute Error:       {median_error:.4f}")
    print(f"{'='*50}")
    
    # Prepare metrics dictionary
    metrics = {
        'test_r2_score': float(r2),
        'test_rmse': float(rmse),
        'test_mse': float(mse),
        'test_mae': float(mae),
        'test_mape': float(mape),
        'test_max_error': float(max_error),
        'test_median_error': float(median_error),
        'n_test_samples': int(len(y_test)),
        'prediction_mean': float(np.mean(y_pred)),
        'prediction_std': float(np.std(y_pred)),
        'actual_mean': float(np.mean(y_test)),
        'actual_std': float(np.std(y_test))
    }
    
    # Save metrics to JSON file
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_output.path}")
    print("="*50)
    
    # Return key metrics
    outputs = namedtuple('Outputs', ['test_r2', 'test_rmse', 'test_mae', 'test_mape'])
    return outputs(
        test_r2=float(r2),
        test_rmse=float(rmse),
        test_mae=float(mae),
        test_mape=float(mape)
    )


# Component metadata for documentation
COMPONENT_METADATA = {
    'data_extraction': {
        'name': 'Data Extraction',
        'description': 'Fetches versioned dataset from DVC remote storage',
        'inputs': ['data_path', 'dvc_remote'],
        'outputs': ['output_data']
    },
    'data_preprocessing': {
        'name': 'Data Preprocessing',
        'description': 'Cleans, scales, and splits data into train/test sets',
        'inputs': ['input_data', 'test_size', 'random_state'],
        'outputs': ['train_data', 'test_data', 'train_samples', 'test_samples', 'n_features']
    },
    'model_training': {
        'name': 'Model Training',
        'description': 'Trains Random Forest model and saves artifacts',
        'inputs': ['train_data', 'n_estimators', 'max_depth', 'min_samples_split', 'random_state'],
        'outputs': ['model_output', 'train_score', 'feature_importance_top3']
    },
    'model_evaluation': {
        'name': 'Model Evaluation',
        'description': 'Evaluates model on test set and saves metrics',
        'inputs': ['test_data', 'model'],
        'outputs': ['metrics_output', 'test_r2', 'test_rmse', 'test_mae', 'test_mape']
    }
}