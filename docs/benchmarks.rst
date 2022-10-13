-----------------
Benchmark results
-----------------

To monitor ``check_shapes`` overhead we regularly run a set of benchmarks. The currently implemented
benchmark is to fit a linear model, using gradient descent.

The actual overhead you observe on your code will depend on:

* How many, and what kind of checks you do. The more checks you do, and the more complicated checks
  you do, the larger your overhead will be.

* How many function calls you make. If you have a few large batches, you will check fewer shapes,
  and your overhead will be smaller, than if you have many small batches.

For more details about the benchmarks, see their
`readme and implementation <https://github.com/GPflow/check_shapes/tree/develop/benchmark>`_.

.. image:: https://gpflow.github.io/check_shapes/benchmark_plots/overhead.png
  :alt: Plots of check_shapes overhead over time.
