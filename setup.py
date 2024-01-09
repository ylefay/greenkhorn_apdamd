import sys
import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="optimal_transport",
    author="Yvann Le Fay",
    description="JAX implementation of the Greenkhorn Algorithm and adaptive primal-dual accelerated mirror descent (APDAMD) Algorithms.",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "jax",
    ],
    long_description_content_type="text/markdown",
    keywords="regularized optimal transport sinkhorn greenkhorn mirror descent",
    license="MIT",
    license_files=("LICENSE",),
)
