This document provides detailed information for developers working on the course recommendation system with usefulness-based reward functions and weight optimization.


## Usefulness-based Approach

The system uses information usefulness as the primary metric for course recommendations, focusing on maximizing the utility of learning outcomes rather than just job applicability.


### Reward Functions
The system implements two main reward function types:

1. **Usefulness-as-Rwd**:
   - Direct information utility as reward signal
   - Focuses purely on learning value
   - Simple and interpretable

2. **Weighted-Usefulness-as-Rwd**:
   - Combines job applicability (beta1) and utility (beta2)
   - Balanced approach between career goals and learning
   - Requires weight optimization

### Weight Optimization Process
1. **Grid Search**:
   - Test different beta1/beta2 combinations
   - Evaluate performance across multiple k values
   - Find optimal weight balance

2. **Evaluation**:
   - Train models with different weights
   - Measure average applicable jobs
   - Compare performance metrics

3. **Visualization**:
   - Generate heatmaps of weight combinations
   - Plot optimization results
   - Save best weights for pipeline use

## Configuration Guide

The system is configured through `UIR/config/run.yaml` (relative to the repository root) with the following parameters:

### Model Configuration
```yaml
model: "dqn"  # or "ppo", "a2c"
total_steps: 500000
eval_freq: 1000
```

### Environment Configuration
```yaml
threshold: 0.8  # Matching threshold
k: 2  # Number of recommendations
baseline: false  # Baseline model
feature: "Weighted-Usefulness-as-Rwd"  # or "Usefulness-as-Rwd"
```

### Weight Configuration
```yaml
model_weights:
  dqn:
    beta1: 0.1  # Weight for job applications
    beta2: 0.9  # Weight for utility
  ppo:
    beta1: 0.1  # Weight for job applications
    beta2: 0.9  # Weight for utility
```

## Results Management

### Directory Structure
```
UIR/
├── results/             # Training results and logs
├── weight/              # Weight optimization results
└── Scripts/             # Core system files
```


## Warning Suppression

The system includes built-in warning suppression for TensorBoard compatibility issues. From the project root, use **`run.py`** as the quiet launcher (it wraps the same pipeline and weight scripts with broad warning filtering):

### Main Runner Options
```bash
# Run main pipeline without warnings
python run.py --mode pipeline

# Run weight optimization without warnings  
python run.py --mode weights

# Run both weight optimization and pipeline
python run.py --mode both
```


## Weight Optimization Workflow

1. **Run Weight Optimization**:
   ```bash
   # Option 1: Main run (recommended)
   python run.py --mode weights
   
   # Option 2: Direct run (may show warnings)
   python UIR/Scripts/weight_optimization.py --config UIR/config/run.yaml
   ```

2. **Review Results**:
   - Check `weight/weight_optimization_results.png`
   - Note best beta1/beta2 values
   - Analyze performance patterns

3. **Update Configuration**:
   - Add optimal weights to `UIR/config/run.yaml`
   - Set appropriate feature type
   - Configure model parameters

4. **Run Main Pipeline**:
   ```bash
   # Option 1: Main run (recommended)
   python run.py --mode pipeline
   
   # Option 2: Direct run (may show warnings)
   python UIR/Scripts/pipeline.py --config UIR/config/run.yaml
   ```



