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
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Stats:

    with_mean: float
    with_std: float
    without_mean: float
    without_std: float
    rel_overhead_mean: float
    rel_overhead_std: float
    abs_overhead_mean: float
    abs_overhead_std: float

    @staticmethod
    def new(df: pd.DataFrame) -> "Stats":
        by_cs = df.groupby("check_shapes")
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

        rel_overhead_mean = (with_mean / without_mean) - 1
        # https://en.wikipedia.org/wiki/Ratio_distribution#Uncorrelated_noncentral_normal_ratio
        rel_overhead_var = (with_mean_sq / without_mean_sq) * (
            (with_var / with_mean_sq) + (without_var / without_mean_sq)
        )
        rel_overhead_std = np.sqrt(rel_overhead_var)

        abs_overhead_mean = with_mean - without_mean
        abs_overhead_var = with_var + without_var
        abs_overhead_std = np.sqrt(abs_overhead_var)

        return Stats(
            with_mean=with_mean,
            with_std=with_std,
            without_mean=without_mean,
            without_std=without_std,
            rel_overhead_mean=rel_overhead_mean,
            rel_overhead_std=rel_overhead_std,
            abs_overhead_mean=abs_overhead_mean,
            abs_overhead_std=abs_overhead_std,
        )
