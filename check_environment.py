#!/usr/bin/env python3
"""
Environment verification script for MOGCN
Checks if all required dependencies are installed and working
"""

def check_environment():
    """Check if MOGCN dependencies are properly installed"""
    
    print("MOGCN Environment Check")
    print("=" * 40)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # List of required packages
    required_packages = {
        'pandas': '1.4.0',
        'numpy': '1.22.0', 
        'scipy': '1.8.0',
        'sklearn': '1.1.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.11.0',
        'plotly': '5.9.0',
        'torch': '1.12.0',
        'networkx': '2.8.0',
        'umap': '0.5.0',
        'lifelines': '0.27.0',
        'wandb': '0.13.0',
        'tqdm': '4.64.0'
    }
    
    optional_packages = {
        'torch_geometric': '2.0.0',
        'snf': '0.2.2'
    }
    
    missing_packages = []
    installed_packages = []
    
    print("\nRequired Packages:")
    print("-" * 20)
    
    for package, min_version in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
                module_name = 'scikit-learn'
            elif package == 'umap':
                import umap
                version = umap.__version__
                module_name = 'umap-learn'
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                module_name = package
            
            print(f"✅ {module_name}: {version}")
            installed_packages.append(package)
            
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
    
    print("\nOptional Packages:")
    print("-" * 18)
    
    for package, min_version in optional_packages.items():
        try:
            if package == 'torch_geometric':
                import torch_geometric
                version = torch_geometric.__version__
            elif package == 'snf':
                import snf
                version = getattr(snf, '__version__', 'installed')
            
            print(f"✅ {package}: {version}")
            
        except ImportError:
            print(f"⚠️  {package}: Not installed (optional)")
    
    # Check CUDA availability if torch is installed
    if 'torch' in installed_packages:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA: Available ({torch.cuda.device_count()} devices)")
            else:
                print("⚠️  CUDA: Not available (CPU mode only)")
        except:
            print("❌ CUDA: Check failed")
    
    print("\n" + "=" * 40)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages installed!")
        
        # Additional checks
        print("\nAdditional Setup:")
        print("-" * 16)
        
        # Check for config file
        import os
        if os.path.exists('config.py'):
            print("✅ config.py: Found")
        else:
            print("⚠️  config.py: Not found (copy from config_example.py)")
        
        # Check for data directory
        if os.path.exists('data/'):
            print("✅ data/: Directory exists")
        else:
            print("⚠️  data/: Directory not found (create and add your data)")
        
        print("\n✅ Environment ready for MOGCN!")
        return True

if __name__ == "__main__":
    check_environment()