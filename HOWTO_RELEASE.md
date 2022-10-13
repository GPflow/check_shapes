# How to make a new `check_shapes` release

1. Bump the version numbers in the `develop` branch, in *BOTH* the
   [`pyproject.toml`](https://github.com/GPflow/check_shapes/blob/develop/pyproject.toml) file
   and the
   [`__init__`](https://github.com/GPflow/check_shapes/blob/develop/check_shapes/__init__.py).

2. Update [RELEASE.md](https://github.com/GPflow/check_shapes/blob/develop/RELEASE.md).
   - Make sure it contains up-to-date release notes for the next release.
     * They should cover all changes, that are visible to library users on the `develop` branch
       since the most recent release.
     * They should make clear to users whether they might benefit from this release and what
       backwards incompatibilities they might face.
   - Make sure the release version matches what you were setting elsewhere.
   - Make a new copy of the template, to prepare for the next release.

3. Create a release PR from `develop` to `main`.
   - **Make a merge commit. DO NOT SQUASH-MERGE.**
   - If you squash-merge, `main` will be *ahead* of develop (by the squash-merge commit). This
     means we’ll end up with merge conflicts at the following release!

4. Go to the [release page on GitHub](https://github.com/GPflow/check_shapes/releases/new) and
   create a release for a tag “v{VERSION}” (e.g., for version 2.1.3 the tag needs to be `v2.1.3`) to
   `main` branch. Copy the release notes into the description field!

5. You are almost done now! Go to https://github.com/GPflow/check_shapes/actions and monitor the
   tests for your newly-created release passed and the job for pushing the pip package succeeded.
   GitHub Actions are triggered by the publication of the release above.

6. Take a break; wait until the new release
   [shows up on PyPi](https://pypi.org/project/check_shapes/#history).


Done done! Go and celebrate our hard work :)
