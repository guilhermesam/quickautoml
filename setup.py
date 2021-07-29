import setuptools

with open("README.md", "r") as docs:
    long_description = docs.read()

setuptools.setup(
    name='firecannon',
    version='0.0.1',
    author='Wabbajack',
    description='Quick tests for supervised machine learning models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=[
        'sklearn',
        'pandas',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
