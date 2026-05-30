from setuptools import setup, find_packages

setup(
    name="gdf_mpm",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "gdal>=3.6",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "geopandas>=0.14",
        "networkx>=3.1",
        "joblib>=1.2",
    ],
    author="Abdallah M. Mohamed Taha",
    description="Graph Deep Forest for Mineral Prospectivity Mapping",
    license="GNU",
)
