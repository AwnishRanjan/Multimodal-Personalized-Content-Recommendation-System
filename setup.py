from setuptools import setup, find_packages

setup(
    name="multimodal-personalized-content-recommendation-system",
    version="0.0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'scikit-learn',
        'transformers',
        'Pillow',
        'opencv-python',
        'matplotlib',
        'seaborn'
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
        ],
    },
    entry_points={
        'console_scripts': [
            'run_pipeline=src.main:main',  # Adjust this according to your main function
        ],
    },
    python_requires='>=3.6',
)
