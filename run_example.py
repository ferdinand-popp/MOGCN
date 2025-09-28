#!/usr/bin/env python3
"""
Example script demonstrating how to run MOGCN with your own data
"""

import os
import sys
from utils import validate_data_format, print_data_summary, setup_directories, create_example_config

def run_mogcn_example():
    """
    Example workflow for running MOGCN with user data
    """
    print("MOGCN Example Workflow")
    print("=" * 40)
    
    # Step 1: Check if configuration exists
    if not os.path.exists("config.py"):
        create_example_config()
        print("\nPlease edit config.py with your data paths and run this script again.")
        return
    
    # Step 2: Load configuration
    try:
        from config import (
            DATA_NAME, PATHS_OMICS, PATH_OVERVIEW, 
            EPOCHS, LEARNING_RATE, K, MU, METRIC, DEVICE
        )
    except ImportError as e:
        print(f"Error loading config: {e}")
        print("Please check your config.py file format.")
        return
    
    # Step 3: Validate data files
    print(f"\nValidating data for {DATA_NAME}...")
    
    missing_files = []
    invalid_files = []
    
    # Check omics files
    for path in PATHS_OMICS:
        if not os.path.exists(path):
            missing_files.append(path)
        elif not validate_data_format(path, "omics"):
            invalid_files.append(path)
    
    # Check clinical file
    if not os.path.exists(PATH_OVERVIEW):
        missing_files.append(PATH_OVERVIEW)
    elif not validate_data_format(PATH_OVERVIEW, "clinical"):
        invalid_files.append(PATH_OVERVIEW)
    
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease check your file paths in config.py")
        return
    
    if invalid_files:
        print("Invalid file format:")
        for f in invalid_files:
            print(f"  - {f}")
        print("\nPlease check the DATA_PREPARATION.md guide")
        return
    
    # Step 4: Print data summary
    print_data_summary(PATHS_OMICS, PATH_OVERVIEW)
    
    # Step 5: Setup output directories
    setup_directories()
    
    # Step 6: Run MOGCN
    print("Starting MOGCN analysis...")
    
    # Import train module and run
    try:
        import train
        
        # Override command line arguments with config values
        sys.argv = [
            'train.py',
            '--data_name', DATA_NAME,
            '--paths_omics'] + PATHS_OMICS + [
            '--path_overview', PATH_OVERVIEW,
            '--epochs', str(EPOCHS),
            '--learningrate', str(LEARNING_RATE),
            '--K', str(K),
            '--mu', str(MU),
            '--metric', METRIC,
            '--device', DEVICE,
            '--newdataset'
        ]
        
        print("Running with configuration:")
        print(f"  Dataset: {DATA_NAME}")
        print(f"  Omics files: {len(PATHS_OMICS)}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Device: {DEVICE}")
        print()
        
        # This would call the main training function
        print("Note: To actually run training, uncomment the line below:")
        print("# train.main()")
        
    except ImportError as e:
        print(f"Error importing train module: {e}")
        return

if __name__ == "__main__":
    run_mogcn_example()