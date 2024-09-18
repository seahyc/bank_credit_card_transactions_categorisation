from bank_credit_card_transactions_categorisation import TransactionClassifier

# Sample transactions and labels
transactions = [
    "-7758 BUS/MRT 442843492",
    "-7758 MR BEAN INT L-FUS",
    "AMAZE* GRAB A-6F22R",
    # Add more transactions
]
labels = [
    "Transport",
    "Food",
    "Transport",
    # Corresponding labels
]

# Initialize the classifier
classifier = TransactionClassifier()

# Train the classifier
classifier.train(transactions, labels)

# New transactions to predict
new_transactions = [
    "-7758 KOUFU PTE LTD",
    "AMAZE* GRAB A-6F2RS",
    "GOOGLE STORAGE",
]

# Predict categories
predictions = classifier.predict(new_transactions)
print("Predicted Categories:", predictions)