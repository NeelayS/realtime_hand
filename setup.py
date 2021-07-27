from setuptools import setup, find_packages
from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"

with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_requires(path=REQUIRE_PATH):
    """
    generates requirements from file path given as REQUIRE_PATH
    """
    for line in open(path).read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


setup(
    name="realtime_hand_3d",
    version="0.1.0",
    description="Real-time had shape and pose in RGB videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="realtime_hand_3d.readthedocs.io/",
    author="Neelay Shah",
    author_email="nstraum1@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=list(get_requires()),
)
