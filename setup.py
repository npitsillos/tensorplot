from setuptools import setup

setup(
    name="Py-Vis",
    version="0.1",
    py_modules=["pyvis"],
    install_requires=[
        "Click"
    ],
    entry_points='''
        [console_scripts]
        pyvis=vis:cli
    ''',
)
