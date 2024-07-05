from setuptools import setup

setup(
    name="mstme",
    version="0.0.1",
    install_requires=[
        "cartopy",
        "matplotlib",
        "numpy",
        "openturns",
        "xarray",
        "dask",
        "pathos",
        "scipy",
        "statsmodels",
        "tqdm",
        "shapely",
    ],
    extras_require={"develop": []},
    entry_points={
        "console_scripts": [],
        "gui_scripts": [],
    },
)
