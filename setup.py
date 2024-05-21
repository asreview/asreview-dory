from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='ASReview-NEMO',
    version='0.1.0',
    description='ASReview New Exciting Models',
    url='https://github.com/asreview/asreview-nemo',
    author='ASReview LAB Developers',
    author_email='j.j.teijema@uu.nl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.8',
    install_requires=[
        'asreview>=1.0',
        'xgboost',
        'sentence_transformers',
        'gensim',
        'scikeras',
        'spacy',
        'scikit-learn',
        'tensorflow',
        'scipy==1.10.1'
    ],
    entry_points={
        'asreview.models.classifiers': [
        ],
        'asreview.models.feature_extraction': [
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview-nemo/issues',
        'Source': 'https://github.com/asreview/asreview-nemo/',
    },
)
