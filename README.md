# check_shapes

`check_shapes` is a library for annotating and checking tensor shapes.
For example:

```python
import tensorflow as tf

from gpflow.experimental.check_shapes import check_shapes

@tf.function
@check_shapes(
    "features: [batch..., n_features]",
    "weights: [n_features]",
    "return: [batch...]",
)
def linear_model(features: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.einsum("...i,i -> ...", features, weights)
```

For more information see our [documentation](https://gpflow.github.io/check_shapes).

## Installation

The recommended way to install `check_shapes` is from pypi:

```bash
pip install check_shapes
```

### From source

To develop `check_shapes`, check it out from GitHub:

```bash
git clone git@github.com:GPflow/check_shapes.git
```

We use [Poetry](https://python-poetry.org/) to install and manage dependencies. Follow their
instructions for how to install Poetry itself. Then:

```bash
cd check_shapes
poetry install
```

To check you installation run our tests:

```bash
poetry run task test
```