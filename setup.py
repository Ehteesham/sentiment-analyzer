import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_descr = f.read()

__version__ = "0.0.0"

REPO_NAME = "sentiment-analyzer"
AUTHOR_USER_NAME = "Ehteesham"
SRC_REPO = "sentimentAnalyzer"
AUTHOR_EMAIL = "ehteshaam44@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for NLP app",
    long_description=long_descr,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)