echo "Installing required modules"
pip install wheel
pip install setuptools

python setup.py bdist_wheel
pip install dist/quickautoml-1.0.0-py3-none-any.whl
