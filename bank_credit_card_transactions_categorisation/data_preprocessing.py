import re
import string
import pandas as pd
import numpy as np

def clean_description(description):
    """
    Cleans the transaction description by removing irrelevant tokens and normalizing text.
    
    Parameters:
        description (str): The transaction description.
    
    Returns:
        str: The cleaned description.
    """
    if pd.isnull(description):
        return ""
    
    # Remove card prefixes and codes (e.g., '-7758', 'AMAZE*')
    description = re.sub(r'^-\d+\s*', '', description)
    description = re.sub(r'AMAZE\*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'-\d{4}\s+', '', description)
    
    # Remove transaction codes and numbers
    description = re.sub(r'\d+', '', description)
    
    # Convert to lowercase
    description = description.lower()
    
    # Remove punctuation
    description = description.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    description = ' '.join(description.split())
    
    return description

def preprocess_transactions(transactions):
    """
    Preprocesses a list of transaction descriptions.
    
    Parameters:
        transactions (list or pd.Series): The list of transaction descriptions.
    
    Returns:
        list: The list of cleaned descriptions.
    """
    return transactions.apply(clean_description)

def handle_missing_values(df, columns):
    """
    Handles missing values in the specified columns of the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data.
        columns (list): List of column names to check for missing values.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df = df.copy()
    for col in columns:
        if df[col].dtype == 'object':
            df[col].fillna('', inplace=True)
        else:
            df[col].fillna(0, inplace=True)
    return df

def normalize_amounts(amounts):
    """
    Normalizes transaction amounts to have zero mean and unit variance.
    
    Parameters:
        amounts (list or pd.Series): The list of transaction amounts.
    
    Returns:
        pd.Series: The normalized amounts.
    """
    mean = amounts.mean()
    std = amounts.std()
    normalized_amounts = (amounts - mean) / std
    return normalized_amounts

def preprocess_dataframe(df, description_column='Description', date_column='Date', amount_column='Amount'):
    """
    Preprocesses the transaction DataFrame by cleaning descriptions, handling missing values,
    and normalizing amounts.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data.
        description_column (str): The column name for descriptions.
        date_column (str): The column name for dates.
        amount_column (str): The column name for amounts.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = df.copy()
    
    # Handle missing values
    df = handle_missing_values(df, [description_column, date_column, amount_column])
    
    # Clean descriptions
    df['Cleaned_Description'] = preprocess_transactions(df[description_column])
    
    # Normalize amounts
    df['Normalized_Amount'] = normalize_amounts(df[amount_column])
    
    # Convert date to datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    return df

def encode_categorical_features(df, categorical_columns):
    """
    Encodes categorical features using one-hot encoding.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data.
        categorical_columns (list): List of categorical column names to encode.
    
    Returns:
        pd.DataFrame: The DataFrame with encoded categorical features.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df

def remove_outliers(amounts, threshold=3):
    """
    Removes outliers from transaction amounts based on the z-score.
    
    Parameters:
        amounts (pd.Series): The transaction amounts.
        threshold (float): The z-score threshold to identify outliers.
    
    Returns:
        pd.Series: The amounts with outliers removed.
    """
    z_scores = np.abs((amounts - amounts.mean()) / amounts.std())
    filtered_amounts = amounts[z_scores < threshold]
    return filtered_amounts

def standardize_text(text_series):
    """
    Standardizes text data by removing accents and special characters.
    
    Parameters:
        text_series (pd.Series): Series containing text data.
    
    Returns:
        pd.Series: The standardized text data.
    """
    import unicodedata
    def remove_accents(text):
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ASCII', 'ignore').decode('utf-8')
        return str(text)
    
    return text_series.apply(remove_accents)