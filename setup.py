from setuptools import setup, find_packages

setup(
    name='quickautoml',
    version='1.0.0',
    author='Wabbajack',
    packages=find_packages(include=['quickautoml']),
    description='Quick tests for supervised machine learning entities',
    license='MIT',
    install_requires=[
        'sklearn', 'numpy', 'pandas', 'matplotlib', 'optuna'
    ],
    test_suite='tests'
)
