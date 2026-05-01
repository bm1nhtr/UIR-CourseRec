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
├── run.py                    # Launcher (optional warning filtering)
├── setup_environment.py      # Environment setup script (pip + requirements.txt)
├── LICENSE                   # MIT license (replace REPLACE_WITH_COPYRIGHT_HOLDER)
├── CITATION.cff              # Software citation metadata (CFF 1.2.0)
├── pyproject.toml            # Poetry dependencies (preferred)
├── poetry.toml               # Poetry virtualenv location preferences
├── poetry.lock               # Optional until generated locally (then commit for reproducible installs)
├── requirements.txt          # Pip-style list (used by setup_environment.py; keep in sync with pyproject)
└── README.md                 # Main documentation
```

## Quick Start

### UIR (Usefulness of Information Reward)

1. **Install with Poetry (recommended)**

   ```bash
   python -m pip install poetry
   poetry install
   ```

   On the first clone, if `poetry.lock` is missing or you changed `pyproject.toml`, run `poetry lock` so installs are fully pinned. That command needs HTTPS access to PyPI you trust (corporate proxies may need extra root certificates).

2. **Configure** the system in `UIR/config/run.yaml` (paths are relative to the project root):

   - Usefulness parameters and weights
   - RL algorithm and training settings
   - Evaluation cadence (`eval_freq`), seeds (`seed`), runs (`nb_runs`)

3. **Run** the pipeline (from the repository root):

   ```bash
   # Quieter launcher (filters common TensorBoard / TF noise)
   poetry run python run.py --mode pipeline

   # Or without Poetry, after activating your venv:
   python run.py --mode pipeline
   ```

   Direct script entry (shows library warnings verbatim—useful for debugging):

   ```bash
   poetry run python UIR/Scripts/pipeline.py --config UIR/config/run.yaml
   ```

4. **Optional — pip bootstrap**

   ```bash
   pip install -r requirements.txt
   ```

   `setup_environment.py` installs from `requirements.txt`; keep that file aligned with `[tool.poetry.dependencies]` whenever you bump dependencies.



## Available Scripts

### Main Scripts

- **`run.py`** — Launcher with optional subprocess output filtering for TensorBoard-related noise.
- **`setup_environment.py`** — Environment setup and package checking (`pip install -r requirements.txt`).
- **`UIR/Scripts/pipeline.py`** — Main training pipeline.
- **`UIR/Scripts/weight_optimization.py`** — Weight optimization for reward functions.

### Usage Examples

```bash
poetry run python run.py --mode pipeline

# Weight optimization only
poetry run python run.py --mode weights

# Full workflow (weights + training)
poetry run python run.py --mode both

# Environment setup (pip path)
setup_environment.py
```

## Dependencies

Declare dependencies in **`pyproject.toml`**; export for pip-centric workflows:

```bash
python -m pip install poetry-plugin-export
poetry export -f requirements.txt --without-hashes -o requirements.txt
```

After exporting, reconcile any intentional constraints with `[tool.poetry.dependencies]` before committing.

Rough stack (versions resolved in lock / export):

**Core ML/RL:** stable-baselines3, gymnasium, scikit-learn, tensorboard, tensorflow  
**Data:** numpy, pandas  
**Plots:** matplotlib, seaborn  
**Config:** PyYAML  
**Utilities:** tqdm

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE). 

## Citation

If you use this repository in academic work, cite via [CITATION.cff](CITATION.cff) (Citation File Format 1.2.0). Update `authors`, `repository-code`, and related fields before release.



## Documentation

For development notes, reward definitions, configuration details, result layout, training, and evaluation, see **`UIR/README_DEVELOPMENT.md`**.
