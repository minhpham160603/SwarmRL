from setuptools import setup, find_packages

setup(
    name="SwarmRL",
    version="1.0",
    description="A package for SwarmRL environment",
    author="MinhPham",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
