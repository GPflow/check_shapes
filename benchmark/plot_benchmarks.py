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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(output_dir: Path) -> None:
    result_dfs = [pd.read_csv(f) for f in output_dir.glob("results_*.csv")]
    results_df = pd.concat(result_dfs, axis="index", ignore_index=True)

    n_columns = 2
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
        line_y_with_means = []
        line_y_with_uppers = []
        line_y_with_lowers = []
        line_y_without_means = []
        line_y_without_uppers = []
        line_y_without_lowers = []
        line_y_overhead_means = []
        line_y_overhead_uppers = []
        line_y_overhead_lowers = []

        for timestamp, timestamp_df in ax_df.groupby("timestamp"):
            by_cs = timestamp_df.groupby("check_shapes")
            mean_by_cs = by_cs.time_s.mean()
            std_by_cs = by_cs.time_s.std().fillna(0.0)
            var_by_cs = by_cs.time_s.var().fillna(0.0)

            with_mean = mean_by_cs[True]
            with_mean_sq = with_mean ** 2
            with_std = std_by_cs[True]
            with_var = var_by_cs[True]
            without_mean = mean_by_cs[False]
            without_mean_sq = without_mean ** 2
            without_std = std_by_cs[False]
            without_var = var_by_cs[False]

            overhead_mean = (with_mean / without_mean) - 1
            # https://en.wikipedia.org/wiki/Ratio_distribution#Uncorrelated_noncentral_normal_ratio
            overhead_var = (with_mean_sq / without_mean_sq) * (
                (with_var / with_mean_sq) + (without_var / without_mean_sq)
            )
            overhead_std = np.sqrt(overhead_var)

            line_xs.append(timestamp)
            line_y_with_means.append(with_mean)
            line_y_with_uppers.append(with_mean + 1.96 * with_std)
            line_y_with_lowers.append(with_mean - 1.96 * with_std)
            line_y_without_means.append(without_mean)
            line_y_without_uppers.append(without_mean + 1.96 * without_std)
            line_y_without_lowers.append(without_mean - 1.96 * without_std)
            line_y_overhead_means.append(100 * overhead_mean)
            line_y_overhead_uppers.append(100 * (overhead_mean + 1.96 * overhead_std))
            line_y_overhead_lowers.append(100 * (overhead_mean - 1.96 * overhead_std))

        ax = axes[i][0]
        (mean_line,) = ax.plot(line_xs, line_y_with_means, label="with check_shapes")
        color = mean_line.get_color()
        ax.fill_between(line_xs, line_y_with_lowers, line_y_with_uppers, color=color, alpha=0.3)
        (mean_line,) = ax.plot(line_xs, line_y_without_means, label="without check_shapes")
        color = mean_line.get_color()
        ax.fill_between(
            line_xs, line_y_without_lowers, line_y_without_uppers, color=color, alpha=0.3
        )
        ax.set_title(ax_name)
        ax.set_ylabel("time / s")
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend()

        ax = axes[i][1]
        (mean_line,) = ax.plot(line_xs, line_y_overhead_means)
        color = mean_line.get_color()
        ax.fill_between(
            line_xs, line_y_overhead_lowers, line_y_overhead_uppers, color=color, alpha=0.3
        )
        ax.set_title(ax_name)
        ax.set_ylabel("% overhead")
        if np.min(line_y_overhead_lowers) >= 0:
            ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="x", labelrotation=30)

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
