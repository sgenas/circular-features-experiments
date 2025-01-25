from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior.

    Parameters
    ----------
    num_base_items : int
        Number of base items in the experiment (e.g., 7 for days of the week).
    num_modifiers : int
        Number of modifiers per base item (e.g., 2 for early/late variants).
    figsize : tuple[float, float], default=(5.5 * 4 / 10, 1.8)
        Width and height of the figure in inches.
    scatter_size_base : int, default=100
        Size of primary scatter points.
    scatter_size_modified : int, default=70
        Size of secondary scatter points.
    alpha_normal : float, default=0.7
        Default opacity for scatter points.
    alpha_decreased : float, default=0.5
        Reduced opacity for specific scatter points.
    fontsize : int, default=4
        Size of annotation text.
    dpi : int, default=300
        Dots per inch for output image.
    flip_axis : bool, default=False
        Whether to invert the y-axis.

    """

    num_base_items: int
    num_modifiers: int
    figsize: tuple[float, float] = (5.5 * 4 / 10, 1.8)
    scatter_size_base: int = 100
    scatter_size_modified: int = 70
    alpha_normal: float = 0.7
    alpha_decreased: float = 0.5
    fontsize: int = 4
    dpi: int = 300
    flip_axis: bool = False
    font_family: str | None = None


def auto_set_limits(
    states_pca: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Set the plot limits automatically based on the PCA-transformed state vectors.
    Used if the xlim and ylim are not set manually.

    Parameters
    ----------
    states_pca : np.ndarray
        PCA-transformed state vectors to plot, shape (n_points, 2).

    """
    margin = 1.2  # Add a 20 % margin to the limits

    max_x = np.max(states_pca[:, 0]) * margin
    min_x = np.min(states_pca[:, 0]) * margin
    max_y = np.max(states_pca[:, 1]) * margin
    min_y = np.min(states_pca[:, 1]) * margin

    return (min_y, max_y), (min_x, max_x)


def setup_plot(
    plot_config: PlotConfig, model_name: str, layer: int, states_pca: np.ndarray
) -> tuple[Figure, Axes]:
    """Initialize and configure the matplotlib plot.

    Parameters
    ----------
    plot_config : PlotConfig
        Configuration for plot appearance.
    model_name : str
        Name of the model being visualized.
    layer : int
        Layer number being visualized.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the created plot.
    """
    if plot_config.font_family is not None:
        plt.rcParams["font.family"] = plot_config.font_family

    fig, ax = plt.subplots(figsize=plot_config.figsize)

    ylim, xlim = auto_set_limits(states_pca)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_title(f"{model_name} - Layer {layer}")

    # Remove spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add center lines
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)

    if plot_config.flip_axis:
        ax.invert_yaxis()

    return fig, ax


def get_point_styles(
    num_base_items: int, num_points: int, num_modifiers: int = 2
) -> list[tuple[float, float, np.ndarray]]:
    """Generate styling information for each scatter point.

    Parameters
    ----------
    num_base_items : int
        Number of base items in the experiment.
    num_points : int
        Total number of points to display
    num_modifiers : int, default=2
        Number of modifiers per base item (e.g., 2 for early/late variants)

    Returns
    -------
    list[tuple[float, float, np.ndarray]]
        List of tuples containing (size, alpha, color) for each point.

    Notes
    -----
    The function generates a consistent color scheme using the twilight colormap.
    It ensures that base items and their modified variants share similar colors.
    Base items (like "Monday" or "January") get larger points, while modified
    versions get smaller points.

    """
    
    # Generate the base colormap
    cmap = plt.cm.twilight(np.linspace(0, 1, num_base_items))
    
    styles = []
    for mod_num in range(num_modifiers + 1):
        for base_num in range(num_base_items):
            size = 100 if mod_num == 0 else 70
            color = cmap[base_num]
            alpha = 0.7
            styles.append((size, alpha, color))
    
    return styles


def plot_points(
    ax: Axes,
    states_pca: np.ndarray,
    display_labels: list[str],
    plot_config: PlotConfig,
) -> None:
    """Plot the scatter points and their labels. The function applies different styles
    (size, color, alpha) to points based on their position in the sequence.

    """
    styles = get_point_styles(
        num_base_items=plot_config.num_base_items,
        num_points=len(display_labels),
        num_modifiers=plot_config.num_modifiers,
    )

    for i, (label, (size, alpha, color)) in enumerate(zip(display_labels, styles)):
        # Plot scatter point
        ax.scatter(
            states_pca[i, 0],
            states_pca[i, 1],
            color=[color],
            s=size,
            alpha=alpha,
            edgecolor="none",
        )

        # Add label
        ax.annotate(
            label, (states_pca[i, 0], states_pca[i, 1]), fontsize=plot_config.fontsize
        )


def create_plot(
    plot_config: PlotConfig,
    states_pca: np.ndarray,
    layer: int,
    model_name: str,
    display_labels: list[str],
    output_path: Path,
) -> None:
    """Create and save a plot of embeddings for a specific layer.The plot is saved
    as a PNG file in the experiment's output directory with the filename format:
    "{model_name.lower()}_{layer}.png"

    Parameters
    ----------
    plot_config : PlotConfig
        Configuration for plot appearance.
    experiment_config : ExperimentConfig
        Configuration for the experiment.
    states_pca : np.ndarray
        PCA-transformed state vectors to plot, shape (n_points, 2).
    layer : int
        Layer number being visualized.
    model_name : str
        Name of the model being visualized.
    display_labels : list[str]
        Labels for each point in the plot.
    output_path : str
        Path to save the output plot

    """
    # Create and setup the plot
    _, ax = setup_plot(plot_config, model_name, layer, states_pca)

    # Plot the points and labels
    plot_points(
        ax=ax,
        states_pca=states_pca,
        display_labels=display_labels,
        plot_config=plot_config,
    )
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=plot_config.dpi,
    )
    plt.close()
