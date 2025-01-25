import json
from pathlib import Path
from typing import Any

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


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Load a JSON file and return its contents"""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(data: dict[str, Any], file_path: Path) -> None:
    """Save data to a JSON file with proper formatting"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def get_experiment_files(
    base_path: Path, model_config: ModelConfig, experiment_config: ExperimentConfig
) -> list[Path]:
    """Get all JSON files for a specific experiment and model"""
    pattern = f"{model_config.name.lower()}_{experiment_config.name}_*.json"
    return list(base_path.glob(pattern))


def merge_json_files(files: list[Path]) -> dict[str, Any]:
    """Merge multiple JSON files into a single dictionary with layers as main keys"""
    merged_data = {}

    # Process each file and organize by layer
    for file_path in sorted(files, key=lambda x: x.name):
        data = load_json_file(file_path)

        # Extract layer number from filename
        layer = int("".join(filter(str.isdigit, file_path.stem.split("layer")[-1])))

        # Remove layer from the data if it exists (since it's now the key in the main dict)
        if "layer" in data:
            del data["layer"]

        # Use layer as the main key
        merged_data[f"layer_{layer}"] = data

    return merged_data


def main():
    # Define base paths
    repo_root = Path(__file__).parent.parent
    output_data_path = repo_root / "media" / "from_cloud_experiments"

    # Define experiments to process
    experiments = [
        COLOUR_CONFIG,
        HSV_COLOUR_CONFIG,
        HSV_COLOUR_TERTIARY_CONFIG,
        WEEKDAY_CONFIG_VERY,
        WEEKDAY_CONFIG_EXTREMELY,
        MONTH_CONFIG,
        MUSICAL_NOTE_CONFIG,
        MUSICAL_NOTE_CONFIG_MOD,
    ]

    # Define models to process
    models = [GEMMA_2B_CONFIG, GEMMA_9B_CONFIG, GEMMA_27B_CONFIG]

    # Process each experiment config for each model config
    for model in models:
        for experiment in experiments:
            # Get all JSON files for this experiment and model
            files = get_experiment_files(
                output_data_path / experiment.data_output_dir / model.name,
                model,
                experiment,
            )
            if not files:
                print(
                    f"No files found for {model.name} - {experiment.name}"
                )
                continue

            # Merge the files
            merged_data = merge_json_files(files)

            # Create output directory if it doesn't exist
            output_dir = output_data_path / "merged_output_data"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save merged data
            output_path = (
                output_dir
                / f"{model.name.lower()}_{experiment.name}_merged.json"
            )
            save_json_file(merged_data, output_path)
            print(f"Merged data saved to {output_path}")


if __name__ == "__main__":
    main()
