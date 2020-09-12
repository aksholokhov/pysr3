cd ..
pip uninstall -y skmixed
rm -rf -y dist/*
python setup.py sdist bdist_wheel
pip install dist/skmixed-*.tar.gz
cd docs || exit
make clean
make html

