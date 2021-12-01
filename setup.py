from setuptools import setup, find_packages

setup(
    name='firecannon',
    version='0.0.1',
    author='Wabbajack',
    packages=find_packages(include=['firecannon']),
    description='Quick tests for supervised machine learning entities',
    license='MIT',
    install_requires=[
        'sklearn', 'numpy', 'pandas', 'matplotlib', 'optuna'
    ],
    test_suite='tests'
)
