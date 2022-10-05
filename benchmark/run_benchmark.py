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
import re
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List, NamedTuple, Sequence, Tuple

import pandas as pd

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S.%f"


class Modifier(NamedTuple):
    """
    A modification to a string.

    The string will be modified by passing these arguments to `re.sub`.
    """

    pattern: str
    repl: str


Modifiers = Tuple[Modifier, ...]


_CHECK_SHAPES_MODIFIER = (
    Modifier(r"@inherit_check_shapes", ""),
    Modifier(r"@check_shapes\(.*?\)", ""),
    Modifier(r"cs\((.*?), \".*?\"\)", r"\1"),
)


_MODIFIERS = {
    "no_compile": (Modifier("@tf.function", ""),),
    "no_jit": (Modifier("@jit", ""),),
}


def run_modified_script(
    script: Path, modifiers: Modifiers, reps: int, keep: bool, output_dir: Path
) -> Sequence[float]:
    modified = output_dir / "tmp.py"
    src = script.read_text()
    for modifier in modifiers:
        src = re.sub(modifier.pattern, modifier.repl, src)
    modified.write_text(src)

    timings = []
    for _ in range(reps):
        before = perf_counter()
        subprocess.run(["python", str(modified)], check=True)
        after = perf_counter()
        t = after - before
        timings.append(t)

    if not keep:
        modified.unlink()

    return timings


def run_benchmark(
    script: Path, reps: int, modifier_strs: List[str], keep: bool, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.is_dir()

    timestamp = datetime.utcnow()
    timestamp_str = timestamp.strftime(TIMESTAMP_FORMAT)

    name_tokens = [script.stem] + modifier_strs
    name = "_".join(name_tokens)

    shared_data = {
        "timestamp": timestamp,
        "script": script.name,
        "modifiers": ",".join(modifier_strs),
        "name": name,
    }

    modifiers = tuple(m for ms in modifier_strs for m in _MODIFIERS[ms])
    with_timings = run_modified_script(script, modifiers, reps, keep, output_dir)
    with_df = pd.DataFrame(
        {
            **shared_data,
            "check_shapes": True,
            "time_s": with_timings,
        }
    )

    modifiers = _CHECK_SHAPES_MODIFIER + modifiers
    without_timings = run_modified_script(script, modifiers, reps, keep, output_dir)
    without_df = pd.DataFrame(
        {
            **shared_data,
            "check_shapes": False,
            "time_s": without_timings,
        }
    )
    df = pd.concat([with_df, without_df], axis="index", ignore_index=True)
    print(df)
    csv_path = output_dir / f"results_{name}_{timestamp_str}.csv"
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modify a script, then times its execution.")
    parser.add_argument("script", type=Path, help="Path to script to modify and run.")
    parser.add_argument("output_dir", type=Path, help="Where to read/write results.")
    parser.add_argument(
        "--reps", type=int, default=5, help="How many times to repeat the benchmark."
    )
    parser.add_argument(
        "--modifiers", nargs="*", default=[], help="Further modifiers to apply to the script."
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete the modified script. Useful for debugging.",
    )
    args = parser.parse_args()

    run_benchmark(args.script, args.reps, args.modifiers, args.keep, args.output_dir)


if __name__ == "__main__":
    main()
