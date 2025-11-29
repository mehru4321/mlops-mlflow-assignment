"""
Script to download and prepare Boston Housing dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os

def download_boston_housing():
    """Download Boston Housing dataset and save as CSV"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Fetch Boston Housing dataset from OpenML
    print("Downloading Boston Housing dataset...")
    boston = fetch_openml(name='boston', version=1, parser='auto')
    
    # Create DataFrame
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    
    # Save to CSV
    output_path = 'data/raw_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())

if __name__ == "__main__":
    download_boston_housing()