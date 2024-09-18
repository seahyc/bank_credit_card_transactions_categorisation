from setuptools import setup, find_packages

setup(
    name='bank_credit_card_transactions_categorisation',
    version='0.1.0',
    description='A package for categorizing bank and credit card transactions.',
    author='Seah Ying Cong',
    author_email='seahyingcong@gmail.com',
    url='https://github.com/seahyc/bank_credit_card_transactions_categorisation',
    packages=find_packages(),
    install_requires=[
        'ocbc_dbs_statement_parser',
        'numpy',
        'pandas',
        'scikit-learn',
        'imbalanced-learn',  # SMOTE is part of this package
        # Add other dependencies if necessary
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6, <=3.8',
)