import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import string
import re

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Load and prepare data
data = pd.read_csv('data/technical_queries.csv')
data['processed_query'] = data['query'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_query'], 
    data['response'], 
    test_size=0.2, 
    random_state=42
)

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
print(f"Training accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test accuracy: {model.score(X_test, y_test):.2f}")

# Save model
with open('models/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

