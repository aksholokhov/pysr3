import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skmixed",
    version="0.0.3",
    author="Aleksei Sholokhov",
    author_email="aksh@uw.edu",
    description="Linear Mixed-Effects Models compatible with SciKit-Learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aksholokhov/dismod_code",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'sklearn', 'pytest']
)
