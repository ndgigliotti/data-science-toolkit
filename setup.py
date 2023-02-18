from setuptools import find_packages, setup

setup(
    name="ndg_tools",
    author="nick_gigliotti",
    author_email="ndgigliotti@gmail.com",
    description="NLP and other DS tools developed for my own use.",
    license="MIT",
    version="0.0.1",
    url="https://github.com/ndgigliotti/data-science-toolkit",
    packages=find_packages(),
    install_requires=[
        "more-itertools",
        "scikit-learn",
        "joblib",
        "numpy",
        "pandas",
        "seaborn",
        "tqdm",
        "nltk",
        "fuzzywuzzy",
        "wordcloud",
    ],
)
