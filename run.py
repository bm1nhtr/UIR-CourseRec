#!/usr/bin/env python3
"""
Silent Runner - Completely suppress all warnings and errors

This script runs the pipeline with complete suppression of TensorBoard warnings.
"""

import os
import sys
import warnings
import subprocess
import re

def suppress_all_warnings():
    """Suppress all possible warnings and errors."""
    
    # Suppress TensorBoard warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_MKL_NATIVE_FORMAT_CONVERTER'] = '0'
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Suppress all warnings
    warnings.simplefilter('ignore')

def filter_tensorboard_warnings(text):
    """Filter out TensorBoard warnings from text."""
    if not text:
        return ""
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip TensorBoard warning lines and empty lines
        if any(warning in line for warning in [
            "ImportError: cannot import name 'notf'",
            "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
            "tensorboard.compat",
            "Traceback (most recent call last):",
            "File \"D:\\Program Files\\Python311\\Lib\\site-packages\\tensorboard\\compat\\__init__.py\"",
            "During handling of the above exception, another exception occurred:",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        ]) or line.strip() == "":
            continue
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def run_pipeline_silent():
    """Run pipeline with complete warning suppression."""
    
    print("Running Course Recommendation Pipeline (Silent Mode)")
    print("All warnings completely suppressed")
    print("=" * 60)
    
    # Change to UIR/Scripts directory
    original_dir = os.getcwd()
    os.chdir("UIR/Scripts")
    
    try:
        # Run pipeline with suppressed output
        result = subprocess.run([
            sys.executable, "pipeline.py", 
            "--config", "../config/run.yaml"
        ], 
        capture_output=True, 
        text=True,
        check=True
        )
        
        print("Pipeline completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        # Filter and show only non-TensorBoard errors
        filtered_stderr = filter_tensorboard_warnings(result.stderr)
        if filtered_stderr.strip():
            print("\nErrors (filtered):")
            print(filtered_stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        # Filter error output too
        filtered_error = filter_tensorboard_warnings(e.stderr)
        if filtered_error.strip():
            print(f"Error output: {filtered_error}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        os.chdir(original_dir)
    
    return True

def run_weight_optimization_silent():
    """Run weight optimization with complete warning suppression."""
    
    print("Running Weight Optimization (Silent Mode)")
    print("All warnings completely suppressed")
    print("=" * 60)
    
    # Change to UIR/Scripts directory
    original_dir = os.getcwd()
    os.chdir("UIR/Scripts")
    
    try:
        # Run weight optimization with suppressed output
        result = subprocess.run([
            sys.executable, "weight_optimization.py", 
            "--config", "../config/run.yaml"
        ], 
        capture_output=True, 
        text=True,
        check=True
        )
        
        print("Weight optimization completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        # Filter and show only non-TensorBoard errors
        filtered_stderr = filter_tensorboard_warnings(result.stderr)
        if filtered_stderr.strip():
            print("\nErrors (filtered):")
            print(filtered_stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Weight optimization failed with error: {e}")
        # Filter error output too
        filtered_error = filter_tensorboard_warnings(e.stderr)
        if filtered_error.strip():
            print(f"Error output: {filtered_error}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        os.chdir(original_dir)
    
    return True

def main():
    """Main function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Course Recommendation System with complete silence")
    parser.add_argument("--mode", choices=["pipeline", "weights", "both"], default="pipeline", 
                       help="What to run: pipeline, weights, or both")
    
    args = parser.parse_args()
    
    print("Course Recommendation System - Silent Mode")
    print("Complete warning suppression")
    print("=" * 60)
    
    # Suppress warnings
    suppress_all_warnings()
    
    success = True
    
    if args.mode == "pipeline":
        success = run_pipeline_silent()
    elif args.mode == "weights":
        success = run_weight_optimization_silent()
    elif args.mode == "both":
        print("1. Running weight optimization first...")
        success1 = run_weight_optimization_silent()
        if success1:
            print("\n2. Running main pipeline...")
            success = run_pipeline_silent()
        else:
            success = False
    
    if success:
        print("\nAll tasks completed successfully!")
        print("No warnings were shown!")
    else:
        print("\nSome tasks failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
