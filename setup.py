from setuptools import setup, find_packages

setup(
    name='pmrf_pipeline',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy~=2.0.2', 'torch~=2.7.0', 'torchvision', 'synapseclient~=4.8.0',
        'nibabel~=5.3.2', 'scipy', 'scikit-image~=0.24.0', 'opencv-python',
        'tqdm~=4.67.1', 'matplotlib~=3.9.4', 'Pillow~=11.2.1', 'PyYAML~=6.0.2',
        'wandb~=0.19.11', 'pandas~=2.2.3', 'pytorch-fid', 'torchio~=0.20.19', 'typer~=0.15.4', 'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'pmrf_pipeline=cli:app',
        ],
    },
)
