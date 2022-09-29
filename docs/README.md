# `check_shapes` documentation

## Read documentation online

The documentation is stored in a special branch
[`gp-pages`](https://github.com/GPflow/check_shapes/tree/gh-pages) and served by
[GitHub Pages](https://pages.github.com/).

We serve a version of documentation for the most recent `main` and `develop` branch pushes.

* `main`: https://gpflow.github.io/check_shapes/main
* `develop`: https://gpflow.github.io/check_shapes/develop

Normally our GitHub Actions are responsible for building our documentation whenever there is a merge
to `develop` or `master`. See the
[configuration](https://github.com/GPflow/check_shapes/blob/develop/.github/workflows) for details.


## Compile documentation locally

To compile the GPflow documentation locally:

```bash
  poetry run task doc
```