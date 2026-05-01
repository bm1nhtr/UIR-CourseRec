"""
Weight Optimization Module for Course Recommendation Pipeline

This module is part of the course recommendation pipeline and is used to find optimal weights
for the weighted reward function in the reinforcement learning environment. It performs grid search
to find the best combination of beta1 (weight for number of applicable jobs) and beta2 (weight for utility).
"""

import os
import warnings

# Suppress TensorBoard and TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _uir_root() -> str:
    """Directory UIR/ (parent of this script's folder)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_root() -> str:
    """Repository root (parent of UIR/)."""
    return os.path.abspath(os.path.join(_uir_root(), ".."))


def adjust_paths_for_current_location(config):
    """Resolve data and results paths from repo root (anchored to __file__, not cwd).
    
    Args:
        config (dict): Configuration dictionary containing file paths
        
    Returns:
        dict: Configuration with absolute paths
    """
    repo_root = _repo_root()
    uir_root = _uir_root()
    path_keys = [
        "taxonomy_path",
        "course_path",
        "cv_path",
        "job_path",
        "mastery_levels_path",
    ]
    for key in path_keys:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(os.path.join(repo_root, config[key]))
    if "results_path" in config:
        config["results_path"] = os.path.join(uir_root, "results")
    return config

"""
Integration with pipeline.py:
-----------------------------
- Grid search over beta1 with beta2 = 1 - beta1 (see WeightOptimizer.grid_search).
  For each combination, trains one RL model per k in {1,2,3}, evaluates mean
  applicable jobs across learners, then averages scores across k.
- Writes plot_results PNG under UIR/weight/weight_optimization_results_<model>.png.
- optimize_weights() merges best beta1/beta2 into the YAML config key model_weights
  for the configured algorithm; pipeline.py reads those weights for CourseRecEnv.

CLI from repository root:
  python UIR/Scripts/weight_optimization.py --config UIR/config/run.yaml

Or: python run.py --mode weights (wrapper around this script).

"""

import numpy as np
from stable_baselines3 import PPO, DQN
from CourseRecEnv import CourseRecEnv
from Dataset import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml
import copy

class WeightOptimizer:
    """
    OOP class for grid search weight optimization, similar to Reinforce.
    Manages all state (dataset, env, model) as instance attributes.
    """
    def __init__(self, dataset, model_name="ppo", threshold=0.8, total_steps=5000):
        self.dataset = dataset
        self.model_name = model_name
        self.threshold = threshold
        self.total_steps = total_steps
        self.k_values = [1, 2, 3]
        # No model cache! Always train new model for each (k, beta1, beta2)

    def train_model(self, k, beta1, beta2):
        print(f"  [TRAIN] Training model for k={k}, beta1={beta1:.2f}, beta2={beta2:.2f} ...")
        dataset_copy_train = copy.deepcopy(self.dataset)
        env_train = CourseRecEnv(
            dataset=dataset_copy_train,
            threshold=self.threshold,
            k=k,
            baseline=False,
            feature="Weighted-Usefulness-as-Rwd",
            beta1=beta1,
            beta2=beta2
        )
        if self.model_name == "dqn":
            model = DQN("MlpPolicy", env_train, verbose=0)
        elif self.model_name == "ppo":
            model = PPO("MlpPolicy", env_train, verbose=0)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        model.learn(total_timesteps=self.total_steps)
        print(f"  [TRAIN] Finished training model for k={k}, beta1={beta1:.2f}, beta2={beta2:.2f}")
        return model

    def evaluate_model(self, model, k, beta1, beta2):
        print(f"  [EVAL] Evaluating model for k={k}, beta1={beta1:.2f}, beta2={beta2:.2f}")
        dataset_copy_eval = copy.deepcopy(self.dataset)
        env_eval = CourseRecEnv(
            dataset=dataset_copy_eval,
            threshold=self.threshold,
            k=k,
            baseline=False,
            feature="Weighted-Usefulness-as-Rwd",
            beta1=beta1,
            beta2=beta2
        )
        total_app_jobs = 0
        n_learners = len(env_eval.dataset.learners)
        for i, learner in enumerate(env_eval.dataset.learners):
            env_eval.reset(learner=learner)
            done = False
            while not done:
                obs = env_eval._get_obs()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env_eval.step(action)
            total_app_jobs += env_eval.dataset.get_nb_applicable_jobs(env_eval._agent_skills, self.threshold)
        avg_app_jobs = total_app_jobs / n_learners
        print(f"    [RESULT] k={k}: avg_applicable_jobs={avg_app_jobs:.4f}")
        return avg_app_jobs

    def grid_search(self):
        beta1_range = np.arange(0.1, 1.0, 0.1)
        results = []
        best_score = float('-inf')
        best_beta1 = None
        best_beta2 = None
        print(f"\n[GRID SEARCH] Starting grid search with {len(beta1_range)} combinations")
        for idx, beta1 in enumerate(beta1_range, 1):
            beta2 = 1 - beta1
            print(f"\n[GRID SEARCH] Testing combination {idx}/{len(beta1_range)}: Beta1={beta1:.2f}, Beta2={beta2:.2f}")
            k_scores = []
            for k in self.k_values:
                model = self.train_model(k, beta1, beta2)
                avg_app_jobs = self.evaluate_model(model, k, beta1, beta2)
                k_scores.append(avg_app_jobs)
            score = np.mean(k_scores)
            results.append((beta1, beta2, score))
            print(f"  [GRID SEARCH] Mean score for beta1={beta1:.2f}, beta2={beta2:.2f}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_beta1 = beta1
                best_beta2 = beta2
                print(f"  [GRID SEARCH] New best combination found!")
        return best_beta1, best_beta2, results

def plot_results(beta1_range, results, model_name):
    """Plot grid search results.
    
    Args:
        beta1_range: Range of beta1 values tested
        results: List of (beta1, score) tuples
        model_name: Name of the model (for output file naming)
    """
    plt.style.use('default')  # Classic white background
    plt.figure(figsize=(10, 6))
    beta1_values = [r[0] for r in results]
    scores = [r[2] for r in results]
    
    plt.plot(beta1_values, scores, 'b-', marker='o')
    plt.xlabel('Beta1 (Beta2 = 1 - Beta1)')
    plt.ylabel('Average Score')
    plt.title('Weight Optimization Results')
    plt.grid(True)
    
    # Add annotations for best point
    best_idx = np.argmax(scores)
    best_x = beta1_values[best_idx]
    best_y = scores[best_idx]
    ylim = plt.ylim()
    y_offset = min(0.05 * (ylim[1] - ylim[0]), ylim[1] - best_y - 0.05 * (ylim[1] - ylim[0]))
    plt.annotate(
        f'Best: ({best_x:.2f}, {best_y:.2f})',
        xy=(best_x, best_y),
        xytext=(10, 30),  # 10 right, 30 up in display coordinates
        textcoords='offset points',
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=1),
        color='black'
    )
    plt.tight_layout()
    
    # Always under <repo>/UIR/weight regardless of cwd
    weight_dir = os.path.join(_uir_root(), "weight")
    os.makedirs(weight_dir, exist_ok=True)
    out_path = os.path.join(weight_dir, f'weight_optimization_results_{model_name}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  [PLOT] Saved {out_path}")

def optimize_weights(config_path):
    """Run weight optimization and update config file.
    
    Args:
        config_path (str): Path to the configuration file
    """
    # Read config file with comments preserved
    with open(config_path, "r") as f:
        config_lines = f.readlines()
        config = yaml.load("".join(config_lines), Loader=yaml.FullLoader)

    # Adjust paths based on current location
    config = adjust_paths_for_current_location(config)

    # Load your dataset
    dataset = Dataset(config)
    
    # Define range for beta1 (beta2 will be 1-beta1)
    beta1_range = np.linspace(0.1, 0.9, 9)  # 9 values from 0.1 to 0.9
    
    # Get model and training parameters from config
    model_name = config.get("model", "ppo")
    threshold = config.get("threshold", 0.8)
    
    # For weight optimization, use fewer timesteps than final training
    final_training_steps = config.get("total_steps", 500000)
    optimization_steps = int(final_training_steps * 0.2)  # Use 20% of final training steps
    
    print(f"\nOptimizing weights for {model_name.upper()}")
    print(f"Grid search: {len(beta1_range)} combinations")
    print(f"Training steps per model: {optimization_steps} (20% of final training steps: {final_training_steps})")
    print("Evaluating across k values: [1, 2, 3]")
    print("Evaluation will be performed on the entire dataset")
    
    optimizer = WeightOptimizer(dataset, model_name, threshold, optimization_steps)
    best_beta1, best_beta2, results = optimizer.grid_search()
    
    # Print results
    print(f"\nBest weights found for {model_name.upper()}:")
    print(f"Beta1: {best_beta1:.2f}")
    print(f"Beta2: {best_beta2:.2f}")
    
    # Plot results
    plot_results(beta1_range, results, model_name)
    
    # Update config with best weights for this model
    if 'model_weights' not in config:
        config['model_weights'] = {}
    
    # Cập nhật weights mới
    config['model_weights'][model_name] = {
        'beta1': float(best_beta1),
        'beta2': float(best_beta2)
    }
    
    # Save updated config while preserving structure and comments
    with open(config_path, "w") as f:
        model_weights_start = -1
        model_weights_end = -1
        in_model_weights = False

        # Find the start and end of the model_weights section
        for i, line in enumerate(config_lines):
            if 'model_weights:' in line:
                model_weights_start = i
                in_model_weights = True
            elif in_model_weights and line.strip() and not line[:1].isspace():
                model_weights_end = i
                in_model_weights = False

        # Write all lines up to model_weights section
        if model_weights_start != -1:
            for i in range(model_weights_start):
                f.write(config_lines[i])
        else:
            for line in config_lines:
                f.write(line)

        # Write the new model_weights section
        f.write('\nmodel_weights:\n')
        for model, weights in config['model_weights'].items():
            f.write(f'  {model}:\n')
            f.write(f'    beta1: {weights["beta1"]}\n')
            f.write(f'    beta2: {weights["beta2"]}\n')

        # Write the remaining lines after the old model_weights section
        if model_weights_end != -1:
            for i in range(model_weights_end, len(config_lines)):
                f.write(config_lines[i])
    
    print(f"\nUpdated config file with best weights for {model_name.upper()}: beta1={best_beta1:.2f}, beta2={best_beta2:.2f}")
    print(f"Note: These weights were found using {optimization_steps} training steps per model.")
    print(f"Final model training will use {final_training_steps} steps.")
    print("These weights are optimized to work well across all k values (1,2,3).")


def resolve_config_path(config_path: str) -> str:
    """Resolve config path from repo root or UIR/config/ (independent of cwd)."""
    if os.path.isabs(config_path):
        return config_path
    if os.path.isfile(config_path):
        return os.path.abspath(config_path)
    at_repo = os.path.normpath(os.path.join(_repo_root(), config_path))
    if os.path.isfile(at_repo):
        return at_repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.normpath(
        os.path.join(script_dir, "..", "config", os.path.basename(config_path))
    )
    if os.path.isfile(sibling):
        return sibling
    return config_path


def main():
    """Main function for the weight optimization pipeline.
    
    This function orchestrates the weight optimization process to find weights
    that work well across all k values (1,2,3).
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default="UIR/config/run.yaml"
    )
    args = parser.parse_args()

    optimize_weights(resolve_config_path(args.config))

if __name__ == "__main__":
    main() 