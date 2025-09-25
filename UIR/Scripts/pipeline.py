import os
import argparse
import yaml
import warnings

# Suppress TensorBoard and TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from Dataset import Dataset
from Reinforce import Reinforce


def adjust_paths_for_current_location(config):
    """Adjust file paths in config based on current working directory.
    
    This function ensures that file paths work whether the script is run from
    the project root (via run_clean.py) or from UIR/Scripts/ directory.
    
    Args:
        config (dict): Configuration dictionary containing file paths
        
    Returns:
        dict: Configuration with adjusted paths
    """
    import os
    
    # Check if we're running from UIR/Scripts/ directory
    current_dir = os.getcwd()
    if current_dir.endswith('UIR/Scripts') or current_dir.endswith('UIR\\Scripts'):
        # We're in UIR/Scripts/, need to go up 2 levels to reach project root
        prefix = "../../"
    else:
        # We're in project root, paths are already correct
        prefix = ""
    
    # Adjust all file paths
    path_keys = ['taxonomy_path', 'course_path', 'cv_path', 'job_path', 'mastery_levels_path']
    for key in path_keys:
        if key in config and not config[key].startswith(prefix):
            config[key] = prefix + config[key]
    
    # Adjust results path
    if 'results_path' in config:
        if current_dir.endswith('UIR/Scripts') or current_dir.endswith('UIR\\Scripts'):
            # We're in UIR/Scripts/, results should be in UIR/results
            config['results_path'] = '../results'
        else:
            # We're in project root, results should be in UIR/results
            config['results_path'] = 'UIR/results'
    
    return config

def create_and_print_dataset(config):
    """Create and initialize the dataset for the recommendation system.
    
    This function creates a Dataset instance using the provided configuration
    and prints its summary information.
    
    Args:
        config (dict): Configuration dictionary containing dataset parameters
        
    Returns:
        Dataset: Initialized dataset object containing learners, jobs, and courses
    """
    # Adjust paths based on current location
    config = adjust_paths_for_current_location(config)
    
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    """Main entry point for the recommendation system pipeline.
    
    This function:
    1. Parses command line arguments to get the configuration file path
    2. Loads the configuration from YAML file
    3. Handles weight optimization if needed
    4. Runs the specified recommendation model for the configured number of iterations
    
    Command line arguments:
        --config: Path to the configuration file (default: "UIR/config/run.yaml")
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default="UIR/config/run.yaml"
    )

    args = parser.parse_args()

    # First load initial config
    with open(args.config, "r") as f:
        initial_config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize beta1 and beta2 as None
    beta1 = None
    beta2 = None

    # Run weight optimization if using weighted reward and weights are not in config
    if initial_config.get("feature") == "Weighted-Usefulness-as-Rwd":
        model_weights = initial_config.get("model_weights", {})
        if initial_config["model"] not in model_weights:
            print(f"\nOptimizing weights for {initial_config['model'].upper()}...")
            from weight_optimization import optimize_weights
            optimize_weights(args.config)
            
            # Reload config after weight optimization
            with open(args.config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = initial_config
            weights = model_weights[initial_config["model"]]
            print(f"\nUsing existing weights for {initial_config['model'].upper()}: beta1={weights['beta1']}, beta2={weights['beta2']}")

        # Get beta values for current model
        model_weights = config.get("model_weights", {})
        current_weights = model_weights.get(config["model"], {})
        beta1 = current_weights.get("beta1")
        beta2 = current_weights.get("beta2")
    else:
        config = initial_config

    for run in range(config["nb_runs"]):
        
        
        dataset = create_and_print_dataset(config)
        
        # Use the Reinforce class for all models
        if config["baseline"]: 
            print("feature: baseline")
            print("-------------------------------------------")
        else: 
            print(f"feature: {config['feature']}")
            print("-------------------------------------------")
            
        recommender = Reinforce(
            dataset,
            config["model"],
            config["k"],
            config["threshold"],
            run,
            config["total_steps"],
            config["eval_freq"],
            config["feature"],
            config["baseline"],
            beta1,
            beta2
        )
        recommender.reinforce_recommendation()

        


if __name__ == "__main__":
    main()