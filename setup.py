from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="decision_infovalue",
    version="0.1.0",
    packages=find_packages(where="decision_infovalue"),
    package_dir={"": "decision_infovalue"},
    python_requires=">=3.8",
    install_requires=requirements,
) 