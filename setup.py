# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='fla',
    version='0.0.1',
    author='Songlin Yang',
    author_email='bestsonta@gmail.com',
    license='MIT',
    description='Fast Triton-based implementations of causal linear attention',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sustcsonglin/flash-linear-attention',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'torch>=2.0',
        'triton',
        'transformers',
        'einops'
    ],
    python_requires='>=3.7',
    zip_safe=False
)
