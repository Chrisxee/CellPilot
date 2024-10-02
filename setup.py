from setuptools import setup, find_packages

setup(
    name="cellpilot",
    version="1.0",
    packages=["cellpilot/"] + find_packages(),
    url="",
    author="Philipp Endres",
    include_package_data=True,
    zip_safe=False,
)