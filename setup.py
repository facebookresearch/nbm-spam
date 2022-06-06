# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools


with open("README.md", encoding="utf8") as f:
    readme = f.read()

DISTNAME = "nbm_spam"
VERSION = "0.1"
DESCRIPTION = "Training and evaluating NBM and SPAM"
LONG_DESCRIPTION = readme
AUTHOR = "Anonymous"

if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        packages=setuptools.find_packages(),
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        setup_requires=["pytest-runner"],
        test_requires=["pytest"],
    )
