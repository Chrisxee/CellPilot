from setuptools import setup, find_packages

setup(
    name="samhi",
    version="1.0",
    packages=["samhi/"] + find_packages(),
    url="",
    author="Philipp Endres",
    include_package_data=True,
    zip_safe=False,
)