import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

class TransactionClassifier:
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, transactions, labels):
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        # Vectorize transaction descriptions
        X = self.vectorizer.fit_transform(transactions)
        # Handle imbalanced data
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # Train the model
        self.model.fit(X_resampled, y_resampled)
        self.is_trained = True

    def update(self, transactions, labels):
        if not self.is_trained:
            raise Exception("Model must be trained before updating.")
        # Encode labels
        y_new = self.label_encoder.transform(labels)
        # Vectorize transaction descriptions
        X_new = self.vectorizer.transform(transactions)
        # Update the model
        self.model.partial_fit(X_new, y_new)
        
    def predict(self, transactions):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        X = self.vectorizer.transform(transactions)
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder
            }, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.label_encoder = data['label_encoder']
            self.is_trained = True