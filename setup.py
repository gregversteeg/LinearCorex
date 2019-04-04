import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="linearcorex",
    version="0.53",
    author="Greg Ver Steeg",
    author_email="gregv at isi.edu",
    description="Linear CorEx finds latent factors that explain relationships in data.",
    long_description_content_type="text/markdown",
    url="https://github.com/gregversteeg/linearcorex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
