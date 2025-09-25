# Reward Shaping for Reinforcement Learning-Based Job-Oriented Course Recommendation

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs using usefulness-based approaches.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It implements a usefulness-based approach that:

- Uses information usefulness as the primary metric
- Implements weighted reward functions
- Focuses on maximizing the utility of course recommendations
- Uses binary skill representation (0-1)

## Project Structure

```
Project/
├── UIR/                      # Usefulness-based approach
│   ├── Scripts/              # Core recommendation system
│   │   ├── CourseRecEnv.py   # RL environment with usefulness
│   │   ├── Reinforce.py      # RL implementation
│   │   ├── Dataset.py        # Data management
│   │   ├── matchings.py      # Skill matching utilities
│   │   └── weight_optimization.py # Weight optimization
│   ├── config/               # Configuration files
│   │   └── run.yaml         # Main configuration
│   ├── results/              # Training results and plots
│   ├── weight/               # Weight optimization results
│   └── README_DEVELOPMENT.md # Detailed development guide
├── Data - Collection/        # Dataset files
│   └── Final/
│       ├── courses.json      # Course data
│       ├── jobs.json         # Job listings
│       ├── resumes.json      # Learner profiles
│       ├── taxonomy.csv      # Skill taxonomy
│       └── mastery_levels.json # Skill mastery definitions
├── run_clean.py              # Clean runner (no warnings)
├── setup_environment.py      # Environment setup script
├── requirements.txt          # Python dependencies
└── README.md                 # Main documentation
```

## Quick Start

### UIR (Usefulness of Information Reward)

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure the system in `config/run.yaml`:
   - Set usefulness parameters and weights
   - Choose RL algorithm and training settings
   - Configure evaluation metrics

3. Run the pipeline:
```bash
# Option 1: Run with complete warning suppression (recommended)
python run.py --mode pipeline

# Option 2: Run directly (may show TensorBoard warnings)
python UIR/Scripts/pipeline.py --config UIR/config/run.yaml
```

### Quick Fix for TensorBoard Warnings

If you see TensorBoard warnings, use one of these solutions:

**Solution 1: Use the main script (recommended)**
```bash
python run.py --mode pipeline
```

**Solution 2: Set environment variables**
```bash
# Windows
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
python UIR/Scripts/pipeline.py --config UIR/config/run.yaml

# Linux/Mac
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
python UIR/Scripts/pipeline.py --config UIR/config/run.yaml
```

**Solution 3: Install compatible packages**
```bash
python setup_environment.py
```

## Available Scripts

The repository includes several utility scripts for different use cases:

### Main Scripts
- **`run.py`** - Main runner with complete warning suppression (recommended)
- **`setup_environment.py`** - Environment setup and package checking
- **`UIR/Scripts/pipeline.py`** - Main training pipeline
- **`UIR/Scripts/weight_optimization.py`** - Weight optimization for reward functions

### Usage Examples
```bash
# Quick start (recommended)
python run.py --mode pipeline

# Weight optimization only
python run.py --mode weights

# Full workflow (weights + training)
python run.py --mode both

# Environment setup
python setup_environment.py
```

## Requirements

### Dependencies

The project requires the following Python packages:

**Core ML/RL Libraries:**
- `stable-baselines3==2.2.1` - Reinforcement learning algorithms (DQN, PPO, A2C)
- `gymnasium>=0.28.0` - RL environment interface (compatible with stable-baselines3)
- `scikit-learn>=1.0.0` - Machine learning utilities (K-means clustering, PCA)
- `tensorboard>=2.13.0` - TensorBoard for logging and visualization
- `tensorflow>=2.13.0` - TensorFlow backend for stable-baselines3

**Data Processing:**
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation and analysis

**Visualization:**
- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization

**Configuration:**
- `PyYAML==6.0.1` - YAML configuration files

**Utilities:**
- `tqdm>=4.62.0` - Progress bars for weight optimization and long-running operations


## Documentation

For detailed information about:
- Development setup and guidelines
- Configuration options
- Results management
- Usefulness-based approach (UIR)
- Model training and evaluation

Please refer to:
- `UIR/README_DEVELOPMENT.md` for usefulness-based approach



