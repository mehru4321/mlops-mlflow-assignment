"""
MLflow-Based Pipeline - Complete Implementation (FIXED)
This orchestrates all pipeline steps using MLflow with proper metric logging
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import json
from datetime import datetime


def step_1_load_data(data_path):
    """
    Step 1: Data Extraction
    Load data from DVC or local source
    """
    print("\n" + "="*70)
    print("STEP 1: DATA EXTRACTION")
    print("="*70)
    
    with mlflow.start_run(run_name="data_extraction", nested=True) as run:
        if os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
        else:
            print("Loading California Housing dataset...")
            housing = fetch_california_housing()
            df = pd.DataFrame(housing.data, columns=housing.feature_names)
            df['target'] = housing.target
            
            # Save for future use
            os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else '.', exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"Data saved to: {data_path}")
        
        # Log metrics
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("total_samples", len(df))
        mlflow.log_param("n_features", len(df.columns) - 1)
        mlflow.log_param("feature_names", ",".join(df.drop('target', axis=1).columns))
        
        # Log as metric to make it visible
        mlflow.log_metric("data_total_samples", len(df))
        mlflow.log_metric("data_n_features", len(df.columns) - 1)
        
        print(f"✓ Data loaded: {df.shape}")
        print(f"  Features: {df.columns.tolist()}")
        
        return df


def step_2_preprocess_data(df, test_size=0.2, random_state=42):
    """
    Step 2: Data Preprocessing
    Clean, scale, and split data
    """
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    
    with mlflow.start_run(run_name="preprocessing", nested=True) as run:
        # Check missing values
        missing_count = df.isnull().sum().sum()
        print(f"Missing values: {missing_count}")
        mlflow.log_param("missing_values", missing_count)
        mlflow.log_metric("preprocessing_missing_values", missing_count)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log as metrics
        mlflow.log_metric("preprocessing_train_samples", len(X_train))
        mlflow.log_metric("preprocessing_test_samples", len(X_test))
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("✓ Features standardized (StandardScaler)")
        mlflow.log_param("scaling_method", "StandardScaler")
        
        # Save scaler
        import joblib
        os.makedirs('artifacts', exist_ok=True)
        scaler_path = 'artifacts/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def step_3_train_model(X_train, y_train, feature_names, n_estimators=100, max_depth=10, random_state=42):
    """
    Step 3: Model Training
    Train Random Forest model
    """
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    with mlflow.start_run(run_name="model_training", nested=True) as run:
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        print(f"Training Random Forest...")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        print(f"✓ Model trained")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("train_rmse", train_rmse)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 3 Important Features:")
        for idx, row in feature_importance.head(3).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        
        # Save feature importance
        importance_path = 'artifacts/feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="HousingPriceModel"
        )
        
        return model


def step_4_evaluate_model(model, X_test, y_test):
    """
    Step 4: Model Evaluation
    Evaluate on test set
    """
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    with mlflow.start_run(run_name="evaluation", nested=True) as run:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        metrics = {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'test_mse': test_mse
        }
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Print results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Test R² Score:    {test_r2:.4f}   (Higher is better, max=1.0)")
        print(f"Test RMSE:        {test_rmse:.4f}   (Lower is better)")
        print(f"Test MAE:         {test_mae:.4f}   (Lower is better)")
        print(f"Test MAPE:        {test_mape:.2f}%  (Lower is better)")
        print("="*70)
        
        # Save metrics to file
        metrics_path = 'artifacts/evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        
        return metrics


def run_complete_pipeline(
    data_path='data/raw/raw_data.csv',
    test_size=0.2,
    n_estimators=100,
    max_depth=10,
    random_state=42
):
    """
    Main pipeline orchestration
    Runs all 4 steps in sequence
    """
    print("\n" + "="*70)
    print("HOUSING PRICE PREDICTION PIPELINE")
    print("MLflow-Based Pipeline Execution")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPipeline Parameters:")
    print(f"  data_path: {data_path}")
    print(f"  test_size: {test_size}")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  random_state: {random_state}")
    
    # Set experiment
    mlflow.set_experiment("housing-price-prediction")
    
    # Start parent run
    with mlflow.start_run(run_name=f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as parent_run:
        
        # Log pipeline parameters
        mlflow.log_param("pipeline_version", "1.0")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Step 1: Load Data
        df = step_1_load_data(data_path)
        
        # Step 2: Preprocess
        X_train, X_test, y_train, y_test, feature_names = step_2_preprocess_data(
            df, test_size, random_state
        )
        
        # Step 3: Train
        model = step_3_train_model(
            X_train, y_train, feature_names, n_estimators, max_depth, random_state
        )
        
        # Step 4: Evaluate
        metrics = step_4_evaluate_model(model, X_test, y_test)
        
        # **CRITICAL FIX**: Log key metrics to parent run so they're visible in main UI
        print("\n📊 Logging metrics to parent run...")
        mlflow.log_metric("final_test_r2", metrics['test_r2'])
        mlflow.log_metric("final_test_rmse", metrics['test_rmse'])
        mlflow.log_metric("final_test_mae", metrics['test_mae'])
        mlflow.log_metric("final_test_mape", metrics['test_mape'])
        mlflow.log_metric("final_test_mse", metrics['test_mse'])
        
        # Also log training info to parent
        mlflow.log_metric("total_samples", len(df))
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Get run info
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nRun ID: {run_id}")
        print(f"\n✓ Metrics logged to parent run (visible in UI)")
        print(f"\nView results in MLflow UI:")
        print(f"  http://localhost:5000")
        print(f"\nTo view this specific run:")
        print(f"  http://localhost:5000/#/experiments/1/runs/{run_id}")
        print("\n💡 TIP: Click on the run in the UI to see all nested runs and detailed metrics")
        print("="*70)
        
        return run_id, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MLflow Housing Price Pipeline')
    parser.add_argument('--data-path', type=str, default='data/raw/raw_data.csv',
                        help='Path to data file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0.0 to 1.0)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum depth of trees')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create artifacts directory
    os.makedirs('artifacts', exist_ok=True)
    
    # Run pipeline
    run_id, metrics = run_complete_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    print("\n✓ All artifacts saved to: artifacts/")
    print("✓ MLflow tracking data saved to: mlruns/")