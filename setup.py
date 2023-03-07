import os
import pathlib
import pkg_resources
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


setuptools.setup(
    name="tempclr",
    version="0.0.1",
    author="Yuncong Yang",
    author_email="yy3035@columbia.edu",
    description="A package for multimodal pretraining.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yyuncong/TempCLR-Dev",
    packages=setuptools.find_packages(),
    install_requires=_read_install_requires(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC-BY-NC",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
