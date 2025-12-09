from setuptools import setup, find_packages
import pathlib

def read_requirements():
    req_file = pathlib.Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return []
    with open(req_file, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="nisqab",
    version="0.1.0",
    description="NISQA audio quality assessment model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "nisqab": ["*.yaml", "weights/*.tar", "sample/*.wav"],
    },
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)