"""
Utility functions for MOGCN data handling and configuration
"""

import os
import sys
from pathlib import Path
import pandas as pd
import warnings

def validate_data_format(filepath, data_type="omics"):
    """
    Validate that data file has the correct format for MOGCN
    
    Args:
        filepath (str): Path to the data file
        data_type (str): Type of data - "omics" or "clinical"
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        df = pd.read_csv(filepath, nrows=5)  # Just check first 5 rows
        
        # Check if first column is Sample_ID
        if df.columns[0] != 'Sample_ID':
            warnings.warn(f"Warning: First column in {filepath} should be 'Sample_ID', found '{df.columns[0]}'")
            return False
            
        # Check for negative values in omics data
        if data_type == "omics":
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[1:]  # Skip Sample_ID
            if len(numeric_cols) > 0:
                has_negatives = (df[numeric_cols] < 0).any().any()
                if has_negatives:
                    warnings.warn(f"Warning: {filepath} contains negative values. All omics values should be positive.")
                    return False
        
        return True
        
    except Exception as e:
        warnings.warn(f"Error reading {filepath}: {str(e)}")
        return False

def check_sample_id_consistency(file_paths):
    """
    Check that all files have consistent Sample_IDs
    
    Args:
        file_paths (list): List of file paths to check
    
    Returns:
        dict: Summary of sample counts and overlaps
    """
    sample_ids = {}
    
    for path in file_paths:
        try:
            df = pd.read_csv(path, usecols=[0])  # Just read first column
            sample_ids[path] = set(df.iloc[:, 0])
        except Exception as e:
            warnings.warn(f"Could not read {path}: {str(e)}")
            sample_ids[path] = set()
    
    # Find intersection of all sample IDs
    all_samples = list(sample_ids.values())
    if all_samples:
        common_samples = set.intersection(*all_samples)
        
        return {
            'common_sample_count': len(common_samples),
            'individual_counts': {path: len(ids) for path, ids in sample_ids.items()},
            'common_samples': common_samples
        }
    else:
        return {'error': 'No valid files found'}

def setup_directories(base_dir="results"):
    """
    Create necessary directories for MOGCN output
    
    Args:
        base_dir (str): Base directory for results
    """
    directories = [
        base_dir,
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "plots"),
        os.path.join(base_dir, "clustering"),
        os.path.join(base_dir, "clinical_analysis")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

def load_config(config_path="config.py"):
    """
    Load configuration from config.py file
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        dict: Configuration parameters
    """
    config = {}
    
    if os.path.exists(config_path):
        # Import config as module
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract configuration variables
        config = {
            attr: getattr(config_module, attr) 
            for attr in dir(config_module) 
            if not attr.startswith('__') and not callable(getattr(config_module, attr))
        }
    else:
        warnings.warn(f"Configuration file {config_path} not found. Using default parameters.")
    
    return config

def print_data_summary(file_paths, clinical_path):
    """
    Print summary of data files for user verification
    
    Args:
        file_paths (list): List of omics data files
        clinical_path (str): Path to clinical data file
    """
    print("\n" + "="*50)
    print("MOGCN Data Summary")
    print("="*50)
    
    print(f"\nOmics Data Files ({len(file_paths)}):")
    for i, path in enumerate(file_paths, 1):
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=1)
            print(f"  {i}. {os.path.basename(path)}")
            print(f"     Samples: {df.shape[0]}, Features: {df.shape[1]-1}")
        else:
            print(f"  {i}. {os.path.basename(path)} - FILE NOT FOUND")
    
    print(f"\nClinical Data File:")
    if os.path.exists(clinical_path):
        df = pd.read_csv(clinical_path, nrows=1)
        print(f"  {os.path.basename(clinical_path)}")
        print(f"  Samples: {df.shape[0]}, Features: {df.shape[1]-1}")
    else:
        print(f"  {os.path.basename(clinical_path)} - FILE NOT FOUND")
    
    # Check sample consistency
    all_files = file_paths + [clinical_path]
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    if len(existing_files) > 1:
        consistency = check_sample_id_consistency(existing_files)
        print(f"\nSample ID Consistency:")
        print(f"  Common samples across all files: {consistency.get('common_sample_count', 'N/A')}")
        
    print("="*50 + "\n")

def create_example_config():
    """
    Create an example configuration file if it doesn't exist
    """
    if not os.path.exists("config.py"):
        print("Creating example config.py file...")
        
        example_config = '''# MOGCN Configuration File
# Copy from config_example.py and modify according to your data

# Dataset configuration
DATA_NAME = "YOUR_DATASET"
DEVICE = "gpu"  # or "cpu"

# Data paths - Update these to your actual file paths
PATHS_OMICS = [
    "data/rnaseq_data.csv",
    "data/methylation_data.csv", 
    "data/cnv_data.csv"
]

PATH_OVERVIEW = "data/clinical_data.csv"

# Training parameters
EPOCHS = 200
LEARNING_RATE = 0.001

# SNF parameters
K = 20
MU = 0.6
METRIC = "sqeuclidean"
'''
        
        with open("config.py", "w") as f:
            f.write(example_config)
            
        print("Created config.py - please edit this file with your data paths!")
        return True
    return False