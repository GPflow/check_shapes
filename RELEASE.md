Release notes for all past releases are available in the ['Releases' section](https://github.com/GPflow/check_shapes/releases) of the `check_shapes` Repo. [HOWTO_RELEASE.md](HOWTO_RELEASE.md) explains just that.

# Release x.y.z (template for future releases)

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

## Breaking Changes

* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API AND BEHAVIORAL BREAKING CHANGES>

## Known Caveats

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM SHOULD GO HERE>

## Major Features and Improvements

* <INSERT MAJOR FEATURE HERE, USING MARKDOWN SYNTAX>
* <IF RELEASE CONTAINS MULTIPLE FEATURES FROM SAME AREA, GROUP THEM TOGETHER>

## Bug Fixes and Other Changes

* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>

## Thanks to our Contributors

This release contains contributions from:

<INSERT>, <NAME>, <HERE>, <USING>, <GITHUB>, <HANDLE>


# Release 1.1.2 (next upcoming release in progress)

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

## Breaking Changes

* <DOCUMENT BREAKING CHANGES HERE>
* <THIS SECTION SHOULD CONTAIN API AND BEHAVIORAL BREAKING CHANGES>

## Known Caveats

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM SHOULD GO HERE>

## Major Features and Improvements

* <INSERT MAJOR FEATURE HERE, USING MARKDOWN SYNTAX>
* <IF RELEASE CONTAINS MULTIPLE FEATURES FROM SAME AREA, GROUP THEM TOGETHER>

## Bug Fixes and Other Changes

* <SIMILAR TO ABOVE SECTION, BUT FOR OTHER IMPORTANT CHANGES / BUG FIXES>
* <IF A CHANGE CLOSES A GITHUB ISSUE, IT SHOULD BE DOCUMENTED HERE>
* <NOTES SHOULD BE GROUPED PER AREA>

## Thanks to our Contributors

This release contains contributions from:

<INSERT>, <NAME>, <HERE>, <USING>, <GITHUB>, <HANDLE>


# Release 1.1.1

Fix missing dependency in `1.1.0`.

## Bug Fixes and Other Changes

* Fix missing dependency on `dropstackframe`.

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 1.1.0

Maintenance release. Mostly importantly adds support for Python 3.11, but also adds a couple of
small features.

## Major Features and Improvements

* Adds support for:
  - Python 3.11
  - PyTorch 2.0
  - TensorFlow 2.12
  - JAX 0.4
* Adds support for annotating `functools.partial` functions.
* Hides `check_shapes` stack frames from stack traces, to be less spammy.

## Thanks to our Contributors

This release contains contributions from:

jesnie, uri-granta


# Release 1.0.0

Release version 1.0.0!

No actual changes from release `0.2.0`.


# Release 0.2.0

This release makes `check_shapes` independent of the underlying framework.

## Major Features and Improvements

* Made `check_shapes` independent of tensor framework:
  - Made NumPy optional.
  - Made TensorFlow optional.
  - Made TensorFlow-Probability optional.
  - Added support for JAX.
  - Added support for PyTorch.
* Added benchmarks and documentation of overhead imposed by `check_shapes`.

## Bug Fixes and Other Changes

* Fixed bug related to `tf.saved_model` and methods wrapped in `@check_shapes`.
* Added support for TensorFlow-Probability `_TensorCoercible` objects.

## Thanks to our Contributors

This release contains contributions from:

jesnie


# Release 0.1.0

Initial import from [GPflow](https://github.com/GPflow/GPflow) and experimental release.

## Thanks to our Contributors

This release contains contributions from:

jesnie
