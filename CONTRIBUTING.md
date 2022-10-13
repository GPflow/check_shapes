# Contributing to `check_shapes`

This file contains notes for potential contributors to `check_shapes`, as well as some notes that may be helpful for maintenance.

## Project scope

The aim of `check_shapes` is to annotate and check the shapes of tensors. There are many other pre- and post conditions one could check, but to keep the scope reasonable we will limit ourselves to tensor shapes.

We welcome contributions to `check_shapes`. If you would like to contribute a feature, please raise discussion via a GitHub issue, to discuss the suitability of the feature within `check_shapes`. If the feature is outside the envisaged scope, we can still link to a separate project in our Readme.

### I have this big feature/extension I would like to add...

Due to limited scope we may not be able to review and merge every feature, however useful it may be. Particularly large contributions or changes to core code are harder to justify against the scope of the project or future development plans. We recommend discussing a possible contribution in an issue before work begins. This should give an indication to how broadly it is supported to bring it into the codebase.

## Code quality requirements

- Code must be covered by tests. We use the [pytest](https://docs.pytest.org/) framework.
- The code must be documented. We use *reST* in docstrings. *reST* is a [standard way of documenting](http://docs.python-guide.org/en/latest/writing/documentation/) in python.
  Missing documentation leads to ambiguities and difficulties in understanding future contributions and use cases.
- Use [type annotations](https://docs.python.org/3/library/typing.html). Type hints make code cleaner and _safer_ to some extent.
- Python code should generally follow the *PEP8* style. We use some custom naming conventions (see below) to have our notation follow the Gaussian process literature. Use `pylint` and `mypy` for formatting and _type checking_.
- Practise writing good code as far as is reasonable. Simpler is usually better. Reading the existing codebase should give a good idea of the expected style.
- `check_shapes` uses [black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) for formatting.
- You can use `poetry run task format_and_test` to check that your code follows the above requirements.

## Pull requests

If you think that your contribution falls within the project scope (see above) please submit a Pull Request (PR) to our GitHub page.
(GitHub provides extensive documentation on [forking](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).)

In order to maintain code quality, and make life easy for the reviewers, please ensure that your PR:

- Only fixes one issue or adds one feature.
- Makes the minimal amount of changes to the existing codebase.
- Is testing its changes.
- Passes all checks (formatting, types, tests - you can run them all locally using `poetry run task test` from the `check_shapes` root directory).

All code goes through a PR; there are no direct commits to the `main` and `develop` branches.

## Version numbering

The main purpose of versioning `check_shapes` is user convenience.

We use the [semantic versioning scheme](https://semver.org/). The semver implies `MAJOR.MINOR.PATCH` version scheme, where `MAJOR` changes when there are incompatibilities in API, `MINOR` means adding functionality without breaking existing API and `PATCH` presumes the code update has backward compatible bug fixes.

When incrementing the version number, this has to be reflected in `./pyproject.toml`.