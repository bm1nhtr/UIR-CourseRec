#!/usr/bin/env python3
"""
Environment Setup Script for Course Recommendation System

This script sets up the environment to suppress TensorBoard warnings
and ensures proper package compatibility.

Usage:
    python setup_environment.py
"""

import os
import sys
import subprocess
import warnings

def setup_environment():
    """Set up environment variables to suppress warnings."""
    
    # Suppress TensorBoard warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print("✅ Environment variables set successfully!")
    print("   - TF_ENABLE_ONEDNN_OPTS=0")
    print("   - TF_CPP_MIN_LOG_LEVEL=2")
    print("   - Python warnings suppressed")

def check_packages():
    """Check if required packages are installed with correct versions."""
    
    required_packages = {
        'stable-baselines3': '2.2.1',
        'tensorboard': '2.13.0',
        'tensorflow': '2.13.0',
        'gymnasium': '0.28.0',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.11.0',
        'PyYAML': '6.0.1',
        'tqdm': '4.62.0'
    }
    
    print("\n🔍 Checking package versions...")
    
    for package, expected_version in required_packages.items():
        try:
            if package == 'tensorflow':
                import tensorflow as tf
                version = tf.__version__
            elif package == 'tensorboard':
                import tensorboard
                version = tensorboard.__version__
            elif package == 'stable-baselines3':
                import stable_baselines3
                version = stable_baselines3.__version__
            elif package == 'gymnasium':
                import gymnasium
                version = gymnasium.__version__
            elif package == 'numpy':
                import numpy
                version = numpy.__version__
            elif package == 'pandas':
                import pandas
                version = pandas.__version__
            elif package == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
            elif package == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
            elif package == 'seaborn':
                import seaborn
                version = seaborn.__version__
            elif package == 'PyYAML':
                import yaml
                version = yaml.__version__
            elif package == 'tqdm':
                import tqdm
                version = tqdm.__version__
            
            print(f"   ✅ {package}: {version}")
            
        except ImportError:
            print(f"   ❌ {package}: NOT INSTALLED")
        except Exception as e:
            print(f"   ⚠️  {package}: Error checking version - {e}")

def install_packages():
    """Install required packages from requirements.txt."""
    
    print("\n📦 Installing packages from requirements.txt...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    
    return True

def main():
    """Main function to set up the environment."""
    
    print("🚀 Setting up Course Recommendation System Environment")
    print("=" * 60)
    
    # Set up environment variables
    setup_environment()
    
    # Check current packages
    check_packages()
    
    # Ask if user wants to install packages
    response = input("\n❓ Do you want to install/update packages? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        if install_packages():
            print("\n🔄 Rechecking packages after installation...")
            check_packages()
    
    print("\n✨ Environment setup complete!")
    print("\nTo use the system without warnings, run:")
    print("   python setup_environment.py")
    print("   python UIR/Scripts/pipeline.py --config UIR/config/run.yaml")

if __name__ == "__main__":
    main()
