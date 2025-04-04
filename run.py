import joblib

# Load the saved model and vectorizer
def load_model():
    model = joblib.load('chatbot_model.pkl')  # Path to your saved model
    vectorizer = joblib.load('vectorizer.pkl')  # Path to your saved vectorizer
    return model, vectorizer

# Predict the reply based on the input message
def predict_reply(input_message, model, vectorizer):
    input_tfidf = vectorizer.transform([input_message])
    reply = model.predict(input_tfidf)[0]
    return reply

# Main function to interact with the user continuously
def main():
    # Load the model and vectorizer
    model, vectorizer = load_model()

    print("Chatbot is ready to talk! Type '/stop' to end the conversation.")
    
    while True:
        # Take user input for the message
        user_input = input("You: ")

        # Check if the user wants to stop the conversation
        if user_input.strip().lower() == "/stop":
            print("Chatbot: Goodbye!")
            break
        
        # Get the reply from the model
        reply = predict_reply(user_input, model, vectorizer)

        # Display the reply
        print(f"Chatbot: {reply}")

if __name__ == "__main__":
    main()
