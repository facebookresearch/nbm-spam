# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools


__version__ = "0.1"

with open("README.md", encoding="utf8") as f:
    readme = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name="nbm_spam",
        version=__version__,
        author="Meta Platforms, Inc.",
        author_email="filipradenovic@fb.com",
        description="Training and evaluating NBM and SPAM",
        long_description=readme,
        url="https://github.com/facebookresearch/nbm-spam",
        license="CC BY-NC 4.0",
        packages=setuptools.find_packages(),
        setup_requires=["pytest-runner"],
        test_requires=["pytest"],
    )
