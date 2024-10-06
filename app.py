from flask import Flask, render_template, request, jsonify, session
import os
import sys

# Assuming the AI model and function are in llama_groq_model.py
sys.path.append('/mnt/data/')
from llama_groq_model import promptGuidelinesFlow

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed to store session data

@app.route("/")
def home():
    # Initialize pastDict for the current session if it's a new conversation
    session['pastDict'] = {}
    return render_template("groq.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["message"]

    # Retrieve session data
    pastDict = session.get('pastDict', {})

    # Hardcoded variables
    tokens = 500
    overlap = 0.1
    path = "allAsha.pdf"

    # Call the chatbot function with the user's current query and updated pastDict
    try:
        response = promptGuidelinesFlow(tokens, overlap, path, pastDict, query=user_input)
        # Update session with the new conversation state (update pastDict)
        session['pastDict'] = pastDict
    except Exception as e:
        response = "Error processing your request. Please try again later."

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
