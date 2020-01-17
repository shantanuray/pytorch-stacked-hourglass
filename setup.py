"""Setup Configuration."""
from setuptools import setup, find_packages


setup(
    name='pytorch-stacked-hourglass',
    version='1.0.0a7',
    description='Stacked Hourglass for Markerless tracking',
    maintainer='Shantanu Ray',
    url='https://github.com/shantanuray/pytorch-stacked-hourglass',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'importlib_resources',
        'Pillow',
        'scipy',
        'tabulate',
        'torch',
    ],
    classifiers=[
        'Topic :: Scientific/Engineering :: Keypoint Detection',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Markerless tracking',
        'Intended Audience :: Behavioral experiments',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ]
)
