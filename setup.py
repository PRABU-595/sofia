from setuptools import setup, find_packages

setup(
    name="sofia",
    version="0.1.0",
    description="Sofia: Same Model, Different Reasons — Quantifying Sampling-Induced Explanation Instability in GNNs",
    author="Sofia Author",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torch-geometric>=2.4",
        "torch-scatter",
        "torch-sparse",
        "captum",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm",
        "scikit-learn",
        "pytest",
        "networkx"
    ],
)
