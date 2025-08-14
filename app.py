import os
from flask import Flask, request, jsonify
import google.generativeai as genai

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Gemini API Configuration ---
api_key = os.getenv("API_KEY")

if not api_key:
    print("ERROR: API_KEY environment variable is not set.")
    # Exit the application immediately if the API key is not found
    # This prevents the server from starting in a broken state.
    exit(1)
else:
    try:
        genai.configure(api_key=api_key)
        print("INFO: Gemini API configured successfully.")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API. Exception: {str(e)}")
        exit(1)

# --- Health Check Endpoint (GET) ---
# This endpoint responds to a GET request, which is what a browser sends.
# It simply confirms that the service is running.
@app.route("/", methods=["GET"])
def health_check():
    """
    A simple health check endpoint to confirm the service is running.
    """
    return "Service is running correctly!", 200

# --- Chatbot Endpoint (POST) ---
# This is the original endpoint for generating a response from the Gemini model.
# It requires a POST request with a JSON body.
@app.route("/", methods=["POST"])
def generate_response():
    """
    Main API endpoint for generating a response from the Gemini model.
    """
    # Simple check to ensure the request has a JSON body.
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.json
        user_prompt = data.get("prompt", "")

        if not user_prompt:
            return jsonify({"error": "Prompt is missing from the request"}), 400

        print(f"INFO: Received prompt: {user_prompt}")

        # The model is initialized here.
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Generate content with the model.
        # This will raise an exception if the API key is invalid or there's a problem with the model.
        response = model.generate_content(user_prompt)

        print(f"INFO: Successfully generated response.")

        # Return the generated text in a JSON response.
        return jsonify({"response": response.text})

    except Exception as e:
        # Catch any errors during the generation process and log them.
        print(f"ERROR: An error occurred during response generation. Exception: {str(e)}")
        return jsonify({"response": "An error occurred while generating the response. Please try again."}), 500
