# Bank Credit Card Transactions Categorisation

A Python package for categorizing bank and credit card transactions using machine learning.

## Features

- Default categories for income and expenses.
- Inference function to categorize new transactions.
- Training function to update the model based on user feedback.
- Handles imbalanced data with class weighting and SMOTE.

## Installation

```bash
pip install bank_credit_card_transactions_categorisation
```

## Usage

```python
from bank_credit_card_transactions_categorisation import TransactionClassifier

# Initialize the classifier
classifier = TransactionClassifier()

# Train the classifier with your data
classifier.train(transactions, labels)

# Predict categories for new transactions
predictions = classifier.predict(new_transactions)
```

### Author

Seah Ying Cong (seahyingcong@gmail.com)

### License

This project is licensed under the MIT License.
