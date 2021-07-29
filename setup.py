from setuptools import setup, find_packages

with open("README.md", "r") as docs:
    long_description = docs.read()

setup(
    name='firecannon',
    version='0.0.1',
    author='Wabbajack',
    packages=find_packages(include=['firecannon']),
    description='Quick tests for supervised machine learning models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=[
        'sklearn', 'numpy', 'pandas', 'matplotlib'
    ],
    test_suite='tests'
)
