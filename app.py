import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Try to get the API key from the environment variables
api_key = os.getenv("API_KEY")

if not api_key:
    # This will show up in the Cloud Run logs if the key is not set
    print("ERROR: API_KEY environment variable is not set.")
else:
    try:
        # Configure the API key. This is where the error likely occurs.
        genai.configure(api_key=api_key)
        print("INFO: Gemini API configured successfully.")
    except Exception as e:
        # If the configuration fails, this will print the specific error
        print(f"ERROR: Failed to configure Gemini API. Exception: {str(e)}")

@app.route("/", methods=["POST"])
def generate_response():
    # This is a basic health check for Cloud Run
    if request.method == "POST":
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.json
        user_prompt = data.get("prompt", "")

        if not user_prompt:
            return jsonify({"error": "Prompt is missing from the request"}), 400

        print(f"INFO: Received prompt: {user_prompt}")

        # The model is initialized inside the try block to catch any model-related errors
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content with the model
        response = model.generate_content(user_prompt)
        
        print(f"INFO: Successfully generated response for prompt: {user_prompt}")

        return jsonify({"response": response.text})

    except Exception as e:
        # This will catch any error during generation and print the specific error to the logs
        print(f"ERROR: An error occurred during response generation. Exception: {str(e)}")
        return jsonify({"response": "An error occurred while generating the response. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
