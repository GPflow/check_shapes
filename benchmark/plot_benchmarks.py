# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .stats import Stats

NDArray = Any


def plot(output_dir: Path) -> None:
    result_dfs = [pd.read_csv(f) for f in output_dir.glob("results_*.csv")]
    results_df = pd.concat(result_dfs, axis="index", ignore_index=True)

    n_columns = 3
    n_rows = len(results_df.name.unique())
    width = 6 * n_columns
    height = 4 * n_rows
    fig, axes = plt.subplots(
        ncols=n_columns,
        nrows=n_rows,
        figsize=(width, height),
        squeeze=False,
        dpi=100,
    )

    for i, (ax_name, ax_df) in enumerate(results_df.groupby("name")):
        line_xs = []
        line_ys = []

        for timestamp, timestamp_df in ax_df.groupby("timestamp"):
            line_xs.append(timestamp)
            line_ys.append(Stats.new(timestamp_df))

        def plot_mean_and_std(
            ax: Axes, prefix: str, *, label: Optional[str] = None, scale: float = 1.0
        ) -> None:
            mean_name = f"{prefix}_mean"
            std_name = f"{prefix}_std"

            # pylint: disable=cell-var-from-loop
            mean: NDArray = np.array([getattr(y, mean_name) for y in line_ys]) * scale
            std: NDArray = np.array([getattr(y, std_name) for y in line_ys]) * scale
            lower: NDArray = mean - 1.96 * std
            upper: NDArray = mean + 1.96 * std

            (mean_line,) = ax.plot(line_xs, mean, label=label)
            color = mean_line.get_color()
            ax.fill_between(line_xs, lower, upper, color=color, alpha=0.3)

            ax.set_title(ax_name)
            ax.tick_params(axis="x", labelrotation=30)
            if np.min(lower) > 0:
                ax.set_ylim(bottom=0.0)

        ax = axes[i][0]
        plot_mean_and_std(ax, "with", label="with check_shapes")
        plot_mean_and_std(ax, "without", label="without check_shapes")
        ax.set_ylabel("time / s")
        ax.legend()

        ax = axes[i][1]
        plot_mean_and_std(ax, "abs_overhead")
        ax.set_ylabel("overhead / s")

        ax = axes[i][2]
        plot_mean_and_std(ax, "rel_overhead", scale=100.0)
        ax.set_ylabel("% overhead")

    fig.tight_layout()
    fig.savefig(output_dir / "overhead.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modify a script, then times its execution.")
    parser.add_argument("output_dir", type=Path, help="Where to read/write results.")
    args = parser.parse_args()

    plot(args.output_dir)


if __name__ == "__main__":
    main()
