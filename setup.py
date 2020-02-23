import setuptools
import fos

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fos",
    version=fos.__version__,
    author="FOS Authors",
    author_email="peter@neurallayer.com",
    description="Deeplearning framework for PyTorch",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/neurallayer/fos",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
