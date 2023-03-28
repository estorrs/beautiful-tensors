from setuptools import setup, find_packages
from setuptools.command.install import install
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    # $ pip install beautiful-tensors
    name='beautiful-tensors',
    version='0.0.1',
    description='Turn tensor operations into aesthetic diagrams. Currently supports PyTorch (Keras/TensorFlow coming soon).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dinglab/beautiful-tensors',
    author='Erik Storrs',
    author_email='estorrs@wustl.edu',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='tensors beautiful diagram pytorch tensorflow matrix operations pictures keras',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'tifffile',
        'torch',
        'torchvision',
        'einops',
        'svgpathtools',
        'matplotlib',
        'scikit-image'
    ],
    include_package_data=True,

    entry_points={
        'console_scripts': [
        ],
    },
)
