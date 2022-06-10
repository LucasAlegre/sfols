from setuptools import setup, find_packages

REQUIRED = ['numpy', 'torch', 'gym', 'cvxpy', 'wandb', 'pymoo', 'seaborn', 'pandas', 'tensorboard']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='reinforcement-learning',
    version='0.1',
    packages=['rl',],
    install_requires=REQUIRED,
    long_description=long_description,
    description='Code for the paper Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer.'
)