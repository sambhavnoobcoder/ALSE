"""
ALSE: Adaptive Learned Segmentation Encoder
Setup script for package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="alse",
    version="3.4.0",
    author="Sambhav Dixit",
    author_email="indosambhav@gmail.com",
    description="Adaptive Learned Segmentation Encoder for End-to-End Tokenization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sambhavnoobcoder/alse",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.23.0",
            "ipywidgets>=8.0.0",
        ],
    },
)
