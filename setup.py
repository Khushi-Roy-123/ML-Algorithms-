from setuptools import find_packages, setup

setup(
    name="jklearn",
    version="0.1.0",
    description="A from-scratch machine learning algorithms library.",
    packages=find_packages(),
    install_requires=["numpy>=1.23"],
)
