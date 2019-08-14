import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

classifiers = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

install_requires = [
    'numpy',
    'matplotlib',
    'scipy',
    'pandas',
    'arviz',
]
tests_require = [
    'mpl_scatter_density',
    'numba',
]
setuptools.setup(
    name="ezmc",
    version="0.0.1",
    author="Eoin Travers",
    author_email="eoin.travers@gmail.com",
    description="Easy Peasy MCMC",
    install_requires=install_requires,
    tests_require=tests_require,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eointravers/ezmc",
    packages=setuptools.find_packages(),
    classifiers=classifiers,
)
