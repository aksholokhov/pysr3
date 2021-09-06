cd ..
pip uninstall -y pysr3
rm -rf -y dist/*
python setup.py sdist bdist_wheel
pip install sphinx_rtd_theme
pip install dist/pysr3-*.tar.gz
cd docs || exit
make clean
rm -rf source/*
sphinx-apidoc --separate -f -o source/ ../src/pysr3
make html

