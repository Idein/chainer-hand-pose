from setuptools import setup, find_packages
import os

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "version.py")).read())

setup(
    name="hand",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version=__version__,
)
