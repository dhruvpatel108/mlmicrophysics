#!/usr/bin/env python3
"""
Setup script for PyTorch Microphysics Emulator

This package provides a scalable, constraint-aware PyTorch-based microphysics emulator
for atmospheric modeling, designed for HPC environments with multi-GPU support.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PyTorch Microphysics Emulator for Atmospheric Modeling"

# Read requirements from conda environment
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements_conda.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="pytorch-microphysics-emulator",
    version="1.0.0",
    description="Scalable, constraint-aware PyTorch-based microphysics emulator for atmospheric modeling",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Dhruv Patel",
    license="MIT",
    url="https://github.com/NCAR/mlmicrophysics",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "dask>=2023.0.0",
        "distributed>=2023.0.0",
        "pyarrow>=10.0.0",
        "tqdm>=4.60.0",
        "wandb>=0.15.0",  # Optional: for experiment tracking
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "hpc": [
            "mpi4py>=3.1.0",  # For distributed training
            "horovod>=0.25.0",  # Alternative distributed training
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "atmospheric science",
        "microphysics",
        "machine learning",
        "pytorch",
        "climate modeling",
        "neural networks",
        "hpc",
        "gpu",
    ],
    entry_points={
        "console_scripts": [
            "train-microphysics=scripts.train_streaming_parallel:main",
            "evaluate-model=scripts.evaluate_model:main",
        ],
    },
) 