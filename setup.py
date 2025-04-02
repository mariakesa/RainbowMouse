from setuptools import setup, find_packages

setup(
    name="rainbow_mouse",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
        # Add others like "scikit-learn", "matplotlib" if needed
    ],
    author="Your Name",
    description="Channel-embedding LFP predictor from ViT embeddings",
)
