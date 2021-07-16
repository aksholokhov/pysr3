import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

docs_require=[
    'sphinx', 'sphinx_rtd_theme'
]

setuptools.setup(
    name="skmixed",
    version="0.1.2",
    author="Aleksei Sholokhov",
    author_email="aksh@uw.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aksholokhov/skmixed",
    license='GPLv3+',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'sklearn', 'pytest', 'pandas'],
)
