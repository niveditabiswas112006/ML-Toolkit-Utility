import ml_utils

def run_agri_analysis_workflow():
    print("Starting the Agricultural Data ML Workflow...")
    
    # Step 1: Data Generation
    print("-> Creating synthetic crop data...")
    crop_data = ml_utils.create_synthetic_crop_data(total_records=300, random_seed=101)
    
    # Step 2: Data Profiling
    ml_utils.inspect_dataset_health(crop_data)
    
    # Step 3: Train-Test Split
    print("-> Partitioning data into training and validation sets...")
    features_train, features_test, target_train, target_test = ml_utils.partition_data_for_modeling(
        dataframe=crop_data, 
        target_col='yield_category', 
        test_ratio=0.20
    )
    
    print(f"Training feature matrix shape: {features_train.shape}")
    print(f"Testing feature matrix shape: {features_test.shape}")
    print(f"Training target array shape: {target_train.shape}")
    
    # Step 4: Data Visualization
    print("\n-> Generating environmental scatter plot...")
    ml_utils.visualize_soil_vs_rainfall(crop_data)
    
    print("\n✅ Agricultural ML Workflow executed successfully!")

if __name__ == "__main__":
    run_agri_analysis_workflow()
