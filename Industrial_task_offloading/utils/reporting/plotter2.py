"""Plotting utilities for comparison experiments."""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


class DITENPlotter2:
    """Plot comparison curves with paper-aligned styles."""

    def __init__(self, save_dir: str = "plots"):
        """Initialize the plotter.

        Args:
            save_dir: Output directory for saved plots.
        """
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
            
        # Hardcoded styles to match the distinct lines in the paper's figures
        self.styles = {
            "Mask MAPPO":         {"color": "#1f77b4", "marker": "v"},  # Blue, Triangle Down
            "MAPPO":        {"color": "#ff7f0e", "marker": "^"},  # Orange, Triangle Up
            "MADDPG":       {"color": "#2ca02c", "marker": "o"},  # Green, Circle
            "Graph-GAT Warmup Mask MAPPO":    {"color": "#00ffff", "marker": "s"},  # Cyan, Square
            "e-ATN-MADDPG":   {"color": "#800080", "marker": "+"},  # Purple, Plus
            "Graph-GAT Mask MAPPO": {"color": "#d62728", "marker": "*"},   # Red, Star (Proposed)
            "Graph-GAT Warmup MAPPO": {"color": "#9467bd", "marker": "D"},
        }

    def plot_training_curve(
        self,
        data_dict: Dict[str, List[float]],
        title: str,
        ylabel: str,
        filename: str,
        diagnostic_headers: Optional[Sequence[str]] = None,
        diagnostic_rows: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        """Plot multiple training curves on the same chart.

        Args:
            data_dict: Mapping of algorithm names to metric histories.
            title: Plot title.
            ylabel: Y-axis label.
            filename: Output filename.
            diagnostic_headers: Optional footer-table column labels.
            diagnostic_rows: Optional compact final-episode diagnostics.
        """
        show_diagnostics = bool(diagnostic_headers and diagnostic_rows)
        if show_diagnostics:
            table_height = max(1.4, 0.32 * (len(diagnostic_rows) + 2))
            figure = plt.figure(figsize=(12, 6 + table_height))
            grid = figure.add_gridspec(
                2,
                1,
                height_ratios=[6.0, table_height],
                hspace=0.35,
            )
            axis = figure.add_subplot(grid[0])
            table_axis = figure.add_subplot(grid[1])
        else:
            figure, axis = plt.subplots(figsize=(10, 6))
            table_axis = None
        
        for algo_name, data in data_dict.items():
            episodes = np.arange(len(data))
            
            # Fetch style if defined, otherwise use matplotlib defaults
            style = self.styles.get(algo_name, {"color": None, "marker": None})
            
            # Space out the markers so they are legible (e.g., plot a marker every 20 episodes)
            markevery = max(1, len(data) // 100)
            
            axis.plot(
                episodes,
                data,
                label=algo_name,
                color=style["color"],
                marker=style["marker"],
                markevery=markevery,
                linewidth=1.4,
                markersize=6,
                alpha=0.9,
            )

        # Formatting matching standard IEEE plots
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlabel("Episodes", fontsize=12)
        axis.set_ylabel(ylabel, fontsize=12)
        
        # Place legend in a standard spot (or outside the plot if it gets crowded)
        axis.legend(loc="best", framealpha=0.9)
        axis.grid(True, linestyle="--", alpha=0.6)

        if table_axis is not None:
            table_axis.axis("off")
            table_axis.set_title(
                "Final episode diagnostics | times: Local / Server / Transfer / "
                "Wait (s/device-task)",
                fontsize=9,
                fontweight="bold",
                loc="left",
                pad=4,
            )
            table = table_axis.table(
                cellText=diagnostic_rows,
                colLabels=diagnostic_headers,
                cellLoc="center",
                colLoc="center",
                colWidths=[0.24, 0.16, 0.16, 0.18, 0.26],
                bbox=[0.0, 0.0, 1.0, 0.90],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8 if len(diagnostic_rows) <= 6 else 7)
            for (row_index, _), cell in table.get_celld().items():
                cell.set_edgecolor("#c7c7c7")
                if row_index == 0:
                    cell.set_facecolor("#eeeeee")
                    cell.set_text_props(fontweight="bold")
                elif row_index % 2 == 0:
                    cell.set_facecolor("#f8f8f8")
        
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = os.path.join(self.save_dir, filename)
        figure.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(figure)
        
        print(f"Saved: {filepath}")
