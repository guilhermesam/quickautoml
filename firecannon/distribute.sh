echo "Installing required modules"
pip install wheel
pip install setuptools

python setup.py bdist_wheel
pip install dist/firecannon-0.0.1-py3-none-any.whl
