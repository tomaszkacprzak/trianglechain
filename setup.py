from setuptools import setup


requirements = [
    "numpy",
    "scipy",
    "matplotlib",
    "scipy",
    "sklearn",
    "tqdm",
    "arviz",
    "ekit",
]

setup(
    name="trianglechain",
    version="0.1.0",
    description="Code for plotting 2D marginal distributions",
    url="https://github.com/tomaszkacprzak/trianglechain",
    author="Tomasz Kacpzak",
    license="GPL-3.0 license",
    packages=["trianglechain"],
    install_requires=requirements,
)
