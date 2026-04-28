from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jklearn",
    version="0.1.0",
    author="Khushi",
    description="A from-scratch machine learning algorithms library for educational purposes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jklearn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.7",
            "scikit-learn>=1.3",
            "pytest>=7.0",
        ],
        "streamlit": [
            "streamlit>=1.32",
        ],
    },
)
