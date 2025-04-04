import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re

file_path = "C:/Users/shant/OneDrive/Desktop/chatbot data/chat_data.json"

# Load the JSON data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Prepare the dataset for training
def prepare_data(data):
    # Extract messages and replies
    messages = [item['message'] for item in data]
    replies = [item['reply'] for item in data]
    return messages, replies

# Preprocess the data to normalize text (lowercase and remove punctuation)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters (e.g., punctuation)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Train the model
def train_model(messages, replies):
    # Preprocess messages and replies
    messages = [preprocess_text(msg) for msg in messages]
    replies = [preprocess_text(reply) for reply in replies]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(messages, replies, test_size=0.2, random_state=42)

    # Convert messages into TF-IDF features with n-grams (bi-grams and tri-grams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')  # Unigrams and Bigrams, remove stopwords
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save the model and vectorizer for later use
    joblib.dump(model, 'chatbot_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return model, vectorizer

# Make a prediction
def predict_reply(input_message, model, vectorizer):
    # Preprocess the input message
    input_message = preprocess_text(input_message)
    
    # Transform the input message using the vectorizer
    input_tfidf = vectorizer.transform([input_message])
    
    # Predict the reply
    reply = model.predict(input_tfidf)[0]
    return reply

# Main function
def main():
    # Load the data from the JSON file
    data = load_data(file_path)  # Replace with your file path

    # Prepare messages and replies
    messages, replies = prepare_data(data)

    # Train the model
    model, vectorizer = train_model(messages, replies)

    # Test the model with a sample input message
    test_message = "Kahan th mre tulip mre bday pe hein?"  # Replace with any message
    reply = predict_reply(test_message, model, vectorizer)
    print(f"Input Message: {test_message}")
    print(f"Predicted Reply: {reply}")

if __name__ == "__main__":
    main()
