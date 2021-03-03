from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = ['numpy']

extras = {'with_gpu_support': ['gemmforge==0.0.202']}

setup(
    name="yateto",
    version="0.1.0",
    license="MIT",
    author="Carsten Uphoff",
    author_email="uphoff@in.tum.com",
    description="A tensor toolbox for discontinuous Galerkin methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["yateto"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=install_requires,
    extras_require=extras,
)
