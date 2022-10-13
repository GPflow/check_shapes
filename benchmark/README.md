# `check_shapes` benchmarking.

The benchmarks are based on running the examples in `benchmark/examples`, then editing the script to
remove all shape checking, and running it again.

Most recent results are shown in our
[documentation](https://gpflow.github.io/check_shapes/benchmarks.html).

To run a benchmark use:

```bash
python benchmark/run_benchmark.py \
    <path to example script> \
    [--modifiers=<other modification to the script>] \
    <output_directory>
```

Then plot the results with:

```bash
python benchmark/plot_benchmarks.py <output_directory>
```

The plotter will plot all results found in the output directory, so your can run `run_benchmark.py`
multiple times. An example of benchmarking all the examples might be:

```bash
# Make sure everything is installed:
poetry install
./poetryenv -r poetryenvs install

# Run all benchmarks:
./poetryenv poetryenvs/np_max run python benchmark/run_benchmark.py benchmark/examples/np_example.py benchmark_results
./poetryenv poetryenvs/tf_max run python benchmark/run_benchmark.py benchmark/examples/tf_example.py benchmark_results
./poetryenv poetryenvs/tf_max run python benchmark/run_benchmark.py benchmark/examples/tf_example.py --modifiers=no_compile benchmark_results
./poetryenv poetryenvs/jax_max run python benchmark/run_benchmark.py benchmark/examples/jax_example.py benchmark_results
./poetryenv poetryenvs/jax_max run python benchmark/run_benchmark.py benchmark/examples/jax_example.py --modifiers=no_jit benchmark_results
./poetryenv poetryenvs/torch_max run python benchmark/run_benchmark.py benchmark/examples/torch_example.py benchmark_results

# Plot results:
poetry run python benchmark/plot_benchmarks.py benchmark_results
```