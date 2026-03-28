"""Package setup for savage22-gpu-histogram."""

from setuptools import setup, find_packages

setup(
    name='savage22-gpu-histogram',
    version='0.1.0',
    description='GPU-accelerated histogram building for LightGBM sparse training',
    author='Savage22',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
    ],
    extras_require={
        'gpu': ['cupy-cuda12x>=13.0'],
        'dev': [
            'pytest>=7.0',
            'pytest-benchmark>=4.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'gpu-hist-bench=benchmark.run_benchmark:main',
            'gpu-hist-profile=benchmark.profile_memory:main',
        ],
    },
)
