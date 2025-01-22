"""Configuration classes for the continuity of time features experiments."""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration for the continuity of time features experiment.
    
    Parameters
    ----------
    name : str
        Full name of the model, for example "Gemma-2-2B".
    short_name : str
        Short name of the model to use in filenames, for example "gemma-2b".
    hf_name : str
        Hugging Face model name, for example "google/gemma-2-2B".
    layers_to_plot : list[int]
        List of layer numbers to plot.
    tokenizer_name : str, optional
        Hugging Face tokenizer name, if different from the model name.
    cache_dir : str
        Directory to cache the model.
    device : str
        Device to use for the model, "auto" for automatic, "cpu" for CPU, "cuda" for GPU.
    
    """

    name: str
    short_name: str
    hf_name: str
    layers_to_plot: list[int]
    tokenizer_name: str | None = None
    cache_dir: str = "./media/MODELS"
    device: str = "auto"

    def __repr__(self) -> str:
        return "\n ModelConfig: \n" + "\n".join(
            f"    {key}: {value}" for key, value in self.__dict__.items()
        )


@dataclass
class ExperimentConfig:
    """Configuration for an experiment of continuity of time features.

    Parameters
    ----------
    name : str
        Name of the experiment, for example "weekday_very".
    base_items : list[str]
        List of input strings to use in the experiment, e.g. days of the week or months
        of the year.
    display_labels : list[str]
        List of display labels corresponding to the input strings, for example "Mon" for
        "Monday" or "Jan" for "January".
    pca_n_base_samples : int
        Number of samples to use for fitting PCA. This is used to only fit the PCA model
        on the strings that are not modified (for example "Monday", "Tuesday", etc,
        excluding the strings like "very early on Monday").
    pca_components : int
        Number of PCA components to use in dimensionality reduction.
    string modifiers : list[str], optional
        List of time modifiers to apply to the input strings, e.g. "morning" or "evening".
    optional_prefix : str
        Optional prefix to add to each input string.
    output_dir : str
        Directory to save output plots.

    """
    name: str
    base_items: list[str]
    display_labels: list[str]
    pca_n_base_samples: int
    pca_components: int = 5
    string_modifiers: list[str] | None = None
    optional_prefix: str = ""
    plot_output_dir: str = "media/output_plots"
    data_output_dir: str = "media/output_data"

    def __repr__(self) -> str:
        return "\n ExperimentConfig: \n" + "\n".join(
            f"    {key}: {value}" for key, value in self.__dict__.items()
        )


# %% Experiment configs ------------------------------------------------------------------

WEEKDAY_CONFIG_VERY = ExperimentConfig(
    name="weekday_very",
    base_items=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    display_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    string_modifiers=["very early on", "very late on"],
    pca_n_base_samples=7,
    plot_output_dir="media/output_plots/weekday_experiment",
    data_output_dir="media/output_data/weekday_experiment",
)

WEEKDAY_CONFIG_EXTREMELY = ExperimentConfig(
    name="weekday_extremely",
    base_items=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    display_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    string_modifiers=["extremely early on", "extremely late on"],
    pca_n_base_samples=7,
    plot_output_dir="media/output_plots/weekday_experiment",
    data_output_dir="media/output_data/weekday_experiment",
)

MONTH_CONFIG = ExperimentConfig(
    name="month",
    base_items=[
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Fall",
        "Winter",
        "Spring",
        "Summer"
    ],
    display_labels=[
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Fall",
        "Winter",
        "Spring",
        "Summer",
    ],
    string_modifiers=["early in", "late in"],
    pca_n_base_samples=16,
    plot_output_dir="media/output_plots/month_experiment",
    data_output_dir="media/output_data/month_experiment",
)

COLOUR_CONFIG = ExperimentConfig(
    name="colour",
    base_items=[
        "violet",
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red",
    ],
    display_labels=[
        "Violet",
        "Blue",
        "Cyan",
        "Green",
        "Yellow",
        "Orange",
        "Red",
    ],
    pca_n_base_samples=7,
    plot_output_dir="media/output_plots/colour_experiment",
    data_output_dir="media/output_data/colour_experiment",
)

HSV_COLOUR_CONFIG = ExperimentConfig(
    name="hsv_colour",
    base_items=[
        "HSV 0 100 100 colour values",    # Red
        "HSV 60 100 100 colour values",   # Yellow
        "HSV 120 100 100 colour values",  # Green
        "HSV 180 100 100 colour values",  # Cyan
        "HSV 240 100 100 colour values",  # Blue
        "HSV 300 100 100 colour values",  # Magenta
        "HSV 360 100 100 colour values"   # Red again..
    ],
    display_labels=[
        "Red",
        "Yellow",
        "Green",
        "Cyan",
        "Blue",
        "Magenta",
        "Red2"
    ],
    pca_n_base_samples=7,
    plot_output_dir="media/output_plots/hsv_colour_experiment",
    data_output_dir="media/output_data/hsv_colour_experiment",
)

HSV_COLOUR_CONFIG = ExperimentConfig(
    name="hsv_colour_one_red",
    base_items=[
        "HSV 0 100 100 colour values",    # Red
        "HSV 60 100 100 colour values",   # Yellow
        "HSV 120 100 100 colour values",  # Green
        "HSV 180 100 100 colour values",  # Cyan
        "HSV 240 100 100 colour values",  # Blue
        "HSV 300 100 100 colour values",  # Magenta
    ],
    display_labels=[
        "Red",
        "Yellow",
        "Green",
        "Cyan",
        "Blue",
        "Magenta",
    ],
    pca_n_base_samples=6,
    plot_output_dir="media/output_plots/hsv_colour_experiment",
    data_output_dir="media/output_data/hsv_colour_experiment",
)

HSV_COLOUR_TERTIARY_CONFIG = ExperimentConfig(
    name="hsv_colour_tertiary",
    base_items=[
        "HSV 0 100 100 colour values",    # Red
        "HSV 30 100 100 colour values",   # Orange
        "HSV 60 100 100 colour values",   # Yellow
        "HSV 90 100 100 colour values",   # Yellow-Green / Chartreuse
        "HSV 120 100 100 colour values",  # Green
        "HSV 150 100 100 colour values",  # Green-Cyan / Spring Green
        "HSV 180 100 100 colour values",  # Cyan
        "HSV 210 100 100 colour values",  # Cyan-Blue / Azure
        "HSV 240 100 100 colour values",  # Blue
        "HSV 270 100 100 colour values",  # Violet
        "HSV 300 100 100 colour values",  # Magenta
        "HSV 330 100 100 colour values",  # Red-Magenta / Rose
    ],
    display_labels=[
        "Red",
        "Orange",
        "Yellow",
        "Yellow-Green",
        "Green",
        "Green-Cyan",
        "Cyan",
        "Cyan-Blue",
        "Blue",
        "Violet",
        "Magenta",
        "Red-Magenta",
    ],
    pca_n_base_samples=12,
    plot_output_dir="media/output_plots/hsv_colour_experiment",
    data_output_dir="media/output_data/hsv_colour_experiment",
)

MUSICAL_NOTE_CONFIG = ExperimentConfig(
    name="musical_note",
    base_items=[
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ],
    display_labels=[
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ],
    pca_n_base_samples=12,
    optional_prefix="musical note ",
    plot_output_dir="media/output_plots/musical_note_experiment",
    data_output_dir="media/output_data/musical_note_experiment",
)

MUSICAL_NOTE_CONFIG_MOD = ExperimentConfig(
    name="musical_note_flat_sharp",
    base_items=[
        "C",
        #"C#",
        "D",
        #"D#",
        "E",
        "F",
        #"F#",
        "G",
        #"G#",
        "A",
        #"A#",
        "B",
    ],
    display_labels=[
        "C",
        #"C#",
        "D",
        #"D#",
        "E",
        "F",
        #"F#",
        "G",
        #"G#",
        "A",
        #"A#",
        "B",
    ],
    string_modifiers=["flat", "sharp"],
    optional_prefix="musical note ",
    pca_n_base_samples=7,
    plot_output_dir="media/output_plots/musical_note_experiment",
    data_output_dir="media/output_data/musical_note_experiment",
)

# %% Model configs -----------------------------------------------------------------------

MISTRAL_CONFIG = ModelConfig(
    name="Mistral-7B",
    short_name="mistral",
    hf_name="mistralai/Mistral-7B-v0.1",
    layers_to_plot=[1, 10, 20, 30],
)

GEMMA_2B_CONFIG = ModelConfig(
    name="Gemma-2-2B",
    short_name="gemma-2b",
    hf_name="google/gemma-2-2B",
    layers_to_plot=[i for i in range(1, 27)],
    device="cpu",
)

GEMMA_9B_CONFIG = ModelConfig(
    name="Gemma-2-9B",
    short_name="gemma-9b",
    hf_name="google/gemma-2-9B",
    layers_to_plot=[i for i in range(1, 43)],
    device="cpu",
)
