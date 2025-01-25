"""
Analyze and visualize temporal embeddings across multiple language models. The script
loads a model, generates prompts for weekdays, and extracts hidden states for each
prompt. The hidden states are then reduced to two dimensions using PCA and visualized
in a scatter plot.

The script is designed to be used with the configuration defined in the `config.py` file
and the plotting functions defined in the `plotting.py` file.

"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

from config import (
    ModelConfig,
    ExperimentConfig,
    COLOUR_CONFIG,
    HSV_COLOUR_CONFIG,
    HSV_COLOUR_TERTIARY_CONFIG,
    WEEKDAY_CONFIG_VERY,
    WEEKDAY_CONFIG_EXTREMELY,
    MONTH_CONFIG,
    MUSICAL_NOTE_CONFIG,
    MUSICAL_NOTE_CONFIG_MOD,
    MISTRAL_CONFIG,
    GEMMA_2B_CONFIG,
    GEMMA_9B_CONFIG,
    GEMMA_27B_CONFIG,
)
from plotting import create_plot, PlotConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s:%(name)s:  %(message)s",
    level=logging.INFO,
    handlers=[  # Both log to file and print in terminal
        logging.FileHandler("run_history.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("continuity_experiments")


def load_model(config: ModelConfig) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a model and its tokenizer."""
    try:
        # Add debug info about available GPUs
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU Memory per device:")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        logger.info(f"Loading model: {config.name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_name if config.tokenizer_name is None else config.tokenizer_name
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_name, device_map=config.device, cache_dir=config.cache_dir
        )
        logger.info(f"Model is on device: {model.device}")
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {config.name}: {str(e)}")


def get_prompts_and_labels(config: ExperimentConfig) -> tuple[list[str], list[str]]:
    """Generate experiment prompts and the corresonding display labels based on the
    experiment configuration.
    """
    experiment_prompts = [
        f"{config.optional_prefix}{base_item}" for base_item in config.base_items
    ]
    display_labels = config.display_labels.copy()

    # Add string modifiers to the prompts and display labels if they exist
    if config.string_modifiers:
        experiment_prompts.extend(
            f"{config.optional_prefix}{modifier} {base_item}"
            for modifier in config.string_modifiers
            for base_item in config.base_items
        )
        display_labels.extend(
            f"{modifier.title()} {label}"
            for modifier in config.string_modifiers
            for label in config.display_labels
        )

    return experiment_prompts, display_labels


def get_hidden_states(
    prompts: list[str], tokenizer: AutoTokenizer, model: AutoModelForCausalLM
) -> np.ndarray:
    """Extract hidden states from the model for given prompts."""
    hidden_states = []
    device = model.device  # Get the model's device

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states.append(
                [  # Output last token hidden state for each layer
                    outputs.hidden_states[layer][0][-1].cpu().numpy()
                    for layer in range(len(outputs.hidden_states))
                ]
            )
            logging.info(
                f"Processed prompt '{prompt}' for {len(outputs.hidden_states)} layers"
            )
    return np.array(hidden_states)


def perform_pca(
    hidden_states: np.ndarray, layer: int, config: ExperimentConfig
) -> tuple[np.ndarray, list[float]]:
    """Perform PCA on hidden states for dimensionality reduction and return the top 2
    components.

    Returns
    -------
    np.ndarray
        Transformed hidden states with reduced dimensionality.
    list[float]
        Explained variance ratio for each component.

    """
    logger.info(f"Performing PCA for layer {layer}")
    pca = PCA(n_components=config.pca_components)
    layer_states = hidden_states[:, layer]

    # Only fit on base samples, excluding the time modifiers
    pca.fit(layer_states[: config.pca_n_base_samples])
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return pca.transform(layer_states)[:, :2], pca.explained_variance_ratio_.tolist()


def save_results_json(
    experiment_name: str,
    model_name: str,
    layer: int,
    display_labels: list[str],
    states_pca: np.ndarray,
    explained_variance: list[float],
    output_path: Path,
) -> None:
    data = {
        "experiment_name": experiment_name,
        "model_config": model_name,
        "layer": layer,
        "display_labels": display_labels,
        "states_pca": states_pca.tolist(),
        "explained_variance": explained_variance,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(data, f)


def run_experiment(
    experiment_config: ExperimentConfig,
    model_config: ModelConfig,
) -> None:
    """Run the experiment for a single model."""
    torch.set_grad_enabled(False)

    # Generate experiment strings
    prompts, display_labels = get_prompts_and_labels(experiment_config)

    # Load model
    tokenizer, model = load_model(model_config)

    try:
        # Get hidden states
        hidden_states = get_hidden_states(prompts, tokenizer, model)

        # Create visualizations for specified layers
        for layer in model_config.layers_to_plot:
            states_pca, explained_variance = perform_pca(
                hidden_states, layer, experiment_config
            )

            logger.info(f"Plotting layer {layer}")

            # Create and save plots
            plot_output_dir = Path(
                experiment_config.plot_output_dir + f"/{model_config.name}"
            )
            plot_output_dir.mkdir(parents=True, exist_ok=True)

            plot_full_path = (
                plot_output_dir
                / f"{model_config.name.lower()}_{experiment_config.name}_layer{layer}.png"
            )
            create_plot(
                plot_config=PlotConfig(
                    num_base_items=len(experiment_config.base_items),
                    num_modifiers=len(experiment_config.string_modifiers)
                    if experiment_config.string_modifiers is not None
                    else 0,
                    font_family="Helvetica",
                ),
                states_pca=states_pca,
                layer=layer,
                model_name=model_config.name,
                display_labels=display_labels,
                output_path=plot_full_path,
            )
            logger.info(f"Saved plot to {plot_full_path}")
            # Save the data as well
            data_output_dir = Path(
                experiment_config.data_output_dir + f"/{model_config.name}"
            )
            data_output_dir.mkdir(parents=True, exist_ok=True)
            data_full_path = (
                data_output_dir
                / f"{model_config.name.lower()}_{experiment_config.name}_layer{layer}.json"
            )
            save_results_json(
                experiment_name=experiment_config.name,
                model_name=model_config.name,
                layer=layer,
                display_labels=display_labels,
                states_pca=states_pca,
                explained_variance=explained_variance,
                output_path=data_full_path,
            )
            logger.info(f"Saved data to {data_full_path}")
    finally:
        # Clean up
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":

    def get_minutes_and_seconds(time: float) -> tuple[int, int]:
        """Convert time in seconds to minutes and seconds."""
        return int(time // 60), int(time % 60)

    # Set up the experiments to run here
    all_experiments = [
        COLOUR_CONFIG,
        HSV_COLOUR_CONFIG,
        HSV_COLOUR_TERTIARY_CONFIG,
        WEEKDAY_CONFIG_VERY,
        WEEKDAY_CONFIG_EXTREMELY,
        MONTH_CONFIG,
        MUSICAL_NOTE_CONFIG,
        MUSICAL_NOTE_CONFIG_MOD,
    ]

    for experiment in all_experiments:
        model_config = GEMMA_2B_CONFIG
        logging.info("=" * 100)  # Separator for each experiment
        logging.info(f"New script run started at {datetime.now()}")
        logging.info(
            f"Running experiment '{experiment}' \n \n with model setup \n '{model_config}' \n"
        )
        start_time = time.time()

        run_experiment(experiment_config=experiment, model_config=model_config)

        minutes, seconds = get_minutes_and_seconds(time.time() - start_time)
        logger.info(
            f"\n \n     Done! 🥳 All in all it took {minutes} minutes and {seconds} seconds for the experiment '{experiment.name} \n'"
        )
