"""Plotting utilities for DITEN experiments."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

class DITENPlotter:
    """Generate comparison charts aligned with the paper's visual style."""

    def __init__(self, save_dir: str = "plots"):
        """Initialize the plotter.

        Args:
            save_dir: Output directory for saved plots.
        """
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set a clean style resembling the paper's formatting
        plt.style.use("seaborn-v0_8-whitegrid")
        
    def plot_training_curve(
        self, data_dict: Dict[str, List[float]], title: str, ylabel: str, filename: str
    ) -> None:
        """Generate a line chart of training curves.

        Args:
            data_dict: Mapping of algorithm names to metric histories.
            title: Plot title.
            ylabel: Y-axis label.
            filename: Output filename.
        """
        plt.figure(figsize=(10, 6))
        
        # Define specific markers and colors to mimic the paper's style
        markers = ["*", "^", "s", "o", "d", "v"]
        colors = ["red", "orange", "blue", "green", "purple", "cyan"]
        
        for series_index, (algo_name, values) in enumerate(data_dict.items()):
            # Plot every 10th marker to avoid cluttering the graph, just like the paper
            plt.plot(
                values,
                label=algo_name,
                marker=markers[series_index % len(markers)],
                markevery=20,
                color=colors[series_index % len(colors)],
                linewidth=1.5,
                markersize=6
            )
            
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully to {save_path}")
        plt.close()

    def plot_device_scaling_bar_chart(
        self,
        devices_list: List[int],
        data_dict: Dict[str, List[float]],
        ylabel: str,
        filename: str,
    ) -> None:
        """Generate a grouped bar chart for scaling experiments.

        Args:
            devices_list: List of device counts (e.g., [5, 10, 15]).
            data_dict: Mapping of algorithm names to average metrics.
            ylabel: Y-axis label.
            filename: Output filename.
        """
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(devices_list))
        num_algos = len(data_dict)
        width = 0.8 / num_algos  # Dynamic bar width
        
        colors = ["blue", "yellow", "purple", "green", "orange", "red"]
        
        for series_index, (algo_name, values) in enumerate(data_dict.items()):
            # Offset each algorithm's bar so they group together nicely
            offset = (series_index - num_algos / 2) * width + width / 2
            plt.bar(
                x + offset,
                values,
                width,
                label=algo_name,
                color=colors[series_index % len(colors)],
            )
            
        plt.xlabel("The number of industrial devices", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(x, devices_list)
        plt.legend(loc="best", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved successfully to {save_path}")
        plt.close()
