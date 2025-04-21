from flask import Flask, render_template, request, jsonify
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

app = Flask(__name__)

# Load model
with open('models/chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    query = request.form.get('query')
    file = request.files.get('file')
    
    if file:
        # Handle file upload logic here
        pass
        
    # Rest of your prediction logic
    try:
        query = request.form.get('query', '')
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        processed_query = preprocess_text(query)
        response = model.predict([processed_query])
        return jsonify({"response": response[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)