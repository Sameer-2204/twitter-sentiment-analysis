from setuptools import setup, find_packages

setup(
    name="twitter_analysis",
    version="0.1",
    packages=find_packages(include=["scripts", "scripts.*"]),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "nltk",
        "wordcloud",
        "lightgbm",
        "tensorflow",
        "transformers"
    ],
)
