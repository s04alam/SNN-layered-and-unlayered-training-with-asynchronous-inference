from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt

from aiexperiments.visualizing.results import ResultKey


def plot_results_in_heatmap(
    results_dict,
    result_label,
    vmin,
    vmax,
    train_timesteps,
    inference_timesteps,
    inference_methods,
    round_to_decimals=2,
    colormap="viridis",
    reverse_color=False,
    log_scale=False,
):
    def extract_heatmap_values(train_timestep: int, from_dict: Dict[ResultKey, float]):
        heatmap = np.zeros((len(inference_methods), len(inference_timesteps)))

        for i, inference_method in enumerate(inference_methods):
            for j, inference_timestep in enumerate(inference_timesteps):
                heatmap[i, j] = from_dict.get(
                    ResultKey(
                        train_timestep,
                        inference_timestep,
                        inference_method[0],
                        inference_method[1],
                        inference_method[2],
                        inference_method[3],
                    ),
                    0,
                )
        if log_scale:
            heatmap = np.log(heatmap)
        return heatmap, inference_methods, inference_timesteps

    fig, axs = plt.subplots(
        1,
        len(train_timesteps),
        figsize=(15, 5),
        sharey=True,
    )

    for ax, train_timestep in zip(axs, train_timesteps):
        heatmap, inference_methods, inference_timesteps = extract_heatmap_values(
            train_timestep, results_dict
        )
        cmap = colormap + "_r" if reverse_color else colormap
        cax = ax.imshow(heatmap, cmap=cmap, vmax=vmax, vmin=vmin)

        ax.set_xticks(np.arange(len(inference_timesteps)))
        ax.set_yticks(np.arange(len(inference_methods)))

        # Overlay accuracy values on each grid cell
        for i in range(len(inference_methods)):
            for j in range(len(inference_timesteps)):
                value = heatmap[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.{round_to_decimals}f}",
                    ha="center",
                    va="center",
                    color="w" if value < 0.85 else "k",
                    fontsize=10,
                )

        # Using just the member name for the y-tick labels
        ax.set_xticklabels(inference_timesteps)
        ax.set_yticklabels(
            [
                (im[0].name, im[1].name if im[1] is not None else "", im[2])
                for im in inference_methods
            ]
        )

        ax.set_xlabel("Inference timestep size")
        ax.set_title(f"Train timestep size: {train_timestep}")

    axs[0].set_ylabel("Inference method and sync barrier")

    # Adjust colorbar position
    cbar_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation="horizontal")
    cbar_label = "Log " + result_label if log_scale else result_label
    cbar.set_label(
        cbar_label, rotation=0, labelpad=10
    )  # Add title "Accuracy" to the colorbar

    plt.tight_layout()
    plt.show()
