cd ..
pip uninstall -y skmixed
rm -rf -y dist/*
python setup.py sdist bdist_wheel
pip install sphinx_rtd_theme
pip install dist/skmixed-*.tar.gz
cd docs || exit
make clean
rm -rf source/*
sphinx-apidoc --separate -f -o source/ ../src/skmixed
make html

