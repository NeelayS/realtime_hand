#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Neelay Shah",
    author_email="nstraum1@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Real-time hand pose and shape estimation in RGB videos",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="realtime_hand_3d",
    name="realtime_hand_3d",
    packages=find_packages(include=["realtime_hand_3d", "realtime_hand_3d.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/NeelayS/realtime_hand_3d",
    version="0.1.0",
    zip_safe=False,
)
