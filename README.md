# Circular Features Experiments

This repository contains code for analyzing how language models represent cyclical and temporal concepts (like days of the week, months, colors in HSV space, and musical notes) in their embedding spaces. It's based on one of the experiments from the paper [Not All Language Model Features Are Linear](https://arxiv.org/abs/2405.14860) by Engels et al, and it's basically just a rewrite of [this script](https://github.com/JoshEngels/MultiDimensionalFeatures/blob/main/sae_multid_feature_discovery/other_circle_points.py) from their paper repo.

## Overview

The project examines how different language models (including Mistral and Gemma variants) encode circular/cyclical features in their hidden states. It analyzes whether these models maintain the circular nature of concepts like:
- Days of the week
- Months of the year
- Colors (both named colors and HSV values)
- Musical notes

The code extracts hidden states from different layers of the models, performs dimensionality reduction using PCA, and visualizes the resulting embeddings to study their geometric properties.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/circular-features-experiments.git
cd circular-features-experiments
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
circular-features-experiments/
├── media/                 # Directory for model cache and outputs
│   ├── MODELS/            # Cached model files
│   ├── output_plots/      # Generated visualization plots
│   └── output_data/       # Saved experimental data
├── scripts/
│   ├── download_models.py    # Script to download required models
│   └── merge_output_jsons.py # Script to merge output files for presentation
├── src/
│   ├── config.py         # Configuration classes and experiment setups
│   ├── continuity_experiments.py # Main experimental logic
│   └── plotting.py       # Visualization functions
└── run_history.log       # Experiment run logs
```

## Usage

1. Configure your experiment in `src/config.py`. The repository comes with several pre-configured experiments:
   - `WEEKDAY_CONFIG_VERY`: Analyzes "very early/late" temporal modifiers with weekdays
   - `MONTH_CONFIG`: Studies months and seasons with "early/late" modifiers
   - `COLOUR_CONFIG`: Examines named colors
   - `HSV_COLOUR_CONFIG`: Studies HSV color space representations
   - `MUSICAL_NOTE_CONFIG`: Analyzes musical note representations

2. Run experiments:
```bash
python -m src.continuity_experiments
```

The script will:
- Load the specified language model
- Generate embeddings for each experimental prompt
- Perform PCA dimensionality reduction
- Create visualizations
- Save results as PNG plot and JSONs

## Configuration

### Model Configuration
Use `ModelConfig` to specify:
- Model name and HuggingFace identifier
- Which layers to analyze
- Device settings (CPU/GPU)
- Cache directory

### Experiment Configuration
Use `ExperimentConfig` to define:
- Base items (e.g., days, months, colors)
- Display labels
- String modifiers (e.g., "early", "late")
- PCA settings
- Output directories

## Output

The experiments generate:
- Scatter plots showing the 2D PCA projections of hidden states
- JSON files containing the raw data and experiment metadata
- Detailed logs of the experimental runs
