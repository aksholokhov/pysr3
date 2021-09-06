from pathlib import Path

from setuptools import setup, find_packages

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / 'src'

    about = {}
    with (src_dir / "pysr3" / "__about__.py").open() as f:
        exec(f.read(), about)

    install_requirements = [t.strip() for t in open("requirements.txt", 'r').readlines()]

    test_requirements = [
        'pytest',
    ]

    doc_requirements = [
        'sphinx',
        'sphinx-rtd-theme'
        'nbconvert',
        'nbformat'
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=about['__long_description__'],
        long_description_content_type="text/markdown",
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),

        python_requires='>=3.7',
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            'docs': doc_requirements,
            'test': test_requirements,
            'dev': [doc_requirements, test_requirements]
        },
        zip_safe=False,
    )
