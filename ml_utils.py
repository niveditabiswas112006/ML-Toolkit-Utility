import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random

def create_synthetic_crop_data(total_records=250, random_seed=77):
    """
    Creates a synthetic dataset for predicting crop yield quality based on environmental factors.
    Useful for testing ML models when real agricultural data is unavailable.
    """
    # Generating a synthetic dataset using sklearn
    data_X, data_y = make_classification(
        n_samples=total_records, 
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2, 
        weights=[0.6, 0.4], # Imbalanced classes for realism
        random_state=random_seed
    )
    
    # Mapping generic features to domain-specific names
    agri_columns = ['soil_ph_level', 'avg_rainfall_mm', 'sunlight_hours', 'nitrogen_content']
    
    crop_df = pd.DataFrame(data_X, columns=agri_columns)
    
    # Shifting values to make them look more realistic (e.g., pH around 6-7)
    crop_df['soil_ph_level'] = (crop_df['soil_ph_level'] * 0.5) + 6.5
    crop_df['avg_rainfall_mm'] = (crop_df['avg_rainfall_mm'] * 20) + 100
    
    # Adding the target variable (0: Normal Yield, 1: High Yield)
    crop_df['yield_category'] = data_y
    
    return crop_df

def inspect_dataset_health(dataframe):
    """
    Utility to quickly profile the dataset and check for any anomalies or missing values.
    """
    print("\n" + "="*35)
    print("🌾 DATASET HEALTH REPORT 🌾")
    print("="*35)
    print(f"Total Rows: {dataframe.shape[0]}")
    print(f"Total Columns: {dataframe.shape[1]}")
    
    print("\n--- Missing Value Count ---")
    print(dataframe.isnull().sum())
    
    print("\n--- Quick Data Sample ---")
    print(dataframe.head(3))
    print("="*35 + "\n")

def partition_data_for_modeling(dataframe, target_col, test_ratio=0.25):
    """
    Separates the target variable and splits the dataset for model evaluation.
    """
    features = dataframe.drop(columns=[target_col])
    labels = dataframe[target_col]
    
    x_tr, x_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_ratio, random_state=42, stratify=labels
    )
    
    return x_tr, x_te, y_tr, y_te

def visualize_soil_vs_rainfall(dataframe):
    """
    Generates and saves a scatter plot to analyze the relationship between 
    soil pH, rainfall, and crop yield.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Custom colors for agriculture theme
    color_map = {0: '#e74c3c', 1: '#2ecc71'} # Red for Normal, Green for High Yield
    labels_map = {0: 'Normal Yield', 1: 'High Yield'}
    
    for category in [0, 1]:
        subset = dataframe[dataframe['yield_category'] == category]
        ax.scatter(
            subset['soil_ph_level'], 
            subset['avg_rainfall_mm'], 
            c=color_map[category], 
            label=labels_map[category],
            alpha=0.8,
            edgecolors='white'
        )
        
    ax.set_title("Impact of Soil pH and Rainfall on Crop Yield", fontsize=14, fontweight='bold')
    ax.set_xlabel("Soil pH Level", fontsize=11)
    ax.set_ylabel("Average Rainfall (mm)", fontsize=11)
    
    ax.legend(title="Yield Type")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Save instead of show, avoiding UI issues
    out_filename = 'agri_yield_analysis.png'
    plt.tight_layout()
    plt.savefig(out_filename, dpi=150)
    print(f"Visualization successfully saved to '{out_filename}'")
