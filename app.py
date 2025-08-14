import os
import json
from flask import Flask, request, jsonify, send_from_directory
from google.generativeai import GenerativeModel
import google.generativeai as genai

# --- API Configuration ---
# Your API key will be automatically provided by the Canvas environment.
# DO NOT add your API key here.
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("API key not found. Please ensure the GEMINI_API_KEY environment variable is set.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# --- Flask App Configuration ---
app = Flask(__name__, static_folder='templates')

@app.route("/")
def serve_index():
    """Serves the main HTML page for the chatbot."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/generate_text", methods=["POST"])
def generate_text():
    """Handles the chatbot's API requests for text generation."""
    try:
        data = request.get_json()
        user_message = data.get("text")
        
        if not user_message:
            return jsonify({"error": "Missing 'text' in request body."}), 400

        # Create a chat session with the model.
        chat_session = model.start_chat(history=[])
        
        # Send the user's message to the model and get the response.
        response = chat_session.send_message(user_message)
        
        # Return the model's response.
        return jsonify({"response": response.text})

    except Exception as e:
        # Log the full error for debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Main Entry Point for Cloud Run ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
