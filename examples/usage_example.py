import os, sys
import pandas as pd
from ocbc_dbs_statement_parser import parse_bank_statement

# Append the parent directory to sys.path
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bank_credit_card_transactions_categorisation import TransactionClassifier
from bank_credit_card_transactions_categorisation.data_preprocessing import preprocess_transactions

# Step 1: Parse the bank statement PDF file
pdf_file_path = os.path.expanduser('~/Documents/Bank Statements/statement.pdf')

# Parse the bank statement
parsed_data = parse_bank_statement(pdf_file_path)

# Extract transactions
transactions = parsed_data['transactions']

# Step 2: Load transactions into a DataFrame
transactions_df = pd.DataFrame(transactions)

print("Parsed Transactions:")
print(transactions_df.head())

# Step 3: Preprocess the transaction descriptions
transactions_df['Cleaned_Description'] = preprocess_transactions(transactions_df['Description'])

print("\nCleaned Descriptions:")
print(transactions_df[['Description', 'Cleaned_Description']].head())

# Step 4: Prepare labels (dummy labels for this example)
labels = ['Miscellaneous'] * len(transactions_df)

# Step 5: Initialize and train the classifier
classifier = TransactionClassifier()
classifier.train(transactions_df, labels)

# Step 6: Predict categories
predictions = classifier.predict(transactions_df)
transactions_df['Predicted_Category'] = predictions

print("\nTransactions with Predicted Categories:")
print(transactions_df[['Date', 'Description', 'Amount', 'Predicted_Category']].head())