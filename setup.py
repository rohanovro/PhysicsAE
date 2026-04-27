from setuptools import setup, find_packages

setup(
    name="physicsae",
    version="1.0.0",
    author="Mahmudul Hasan Rohan",
    description="Physics-Guided Self-Supervised Autoencoder for Bearing Fault Detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "shap>=0.42.0",
        "tqdm>=4.65.0",
    ],
)
