try:
    from skbuild import setup
except ImportError:
    import sys
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

setup(
    name="graphinf",
    version=0.1,
    author="Charles Murphy",
    author_email="charles.murphy.1@ulaval.ca",
    url="https://github.com/charlesmurphy1/graphinf",
    license="MIT",
    description="Package for the analysis of stochastic processes on random graphs.",

    zip_safe=False,
    packages=["graphinf"],
    cmake_args=[],
    cmake_source_dir="_graphinf",
    include_package_data=True,
    exclude_package_data={'': ["__pycache__"]},
    extras_require={"test": ["pytest"]},
    python_requires=">=3.6",
)
