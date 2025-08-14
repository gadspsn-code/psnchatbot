import os
import json
from flask import Flask, request, jsonify
from google.generativeai import GenerativeModel
import google.generativeai as genai

# --- API Configuration ---
# Your API key will be automatically provided by the Canvas environment.
# DO NOT add your API key here.
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    # This will cause the app to fail to start if the key is missing,
    # which is good for debugging deployment issues.
    raise ValueError("API key not found. Please ensure the GEMINI_API_KEY environment variable is set.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# --- Flask App Configuration ---
app = Flask(__name__)

# --- In-memory HTML Front-End ---
# This is the entire HTML code, stored as a Python multi-line string.
# We are doing this to ensure the front-end is deployed correctly with the backend,
# eliminating any potential file path issues.
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Chatbot - Final Version</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background: #f3f4f6;
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-4 bg-gray-100">
    <div class="w-full max-w-2xl bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col h-[80vh] md:h-[90vh]">
        
        <!-- Header -->
        <div class="p-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <h1 class="text-2xl font-bold text-gray-800">Gemini Chatbot</h1>
        </div>

        <!-- Chat Area -->
        <div id="chat-container" class="flex-grow p-6 overflow-y-auto space-y-4 bg-gray-50">
            <!-- Initial message or placeholder -->
            <div class="flex items-start">
                <div class="bg-gray-200 text-gray-800 rounded-2xl p-4 max-w-[80%] shadow-md">
                    Hello! This is the new, updated chatbot. How can I help?
                </div>
            </div>
        </div>

        <!-- Input Area -->
        <div class="p-4 border-t border-gray-200 bg-white">
            <form id="chat-form" class="flex items-center space-x-2">
                <input
                    type="text"
                    id="message-input"
                    class="flex-grow p-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
                    placeholder="Type your message..."
                    autocomplete="off"
                />
                <button
                    type="submit"
                    class="p-3 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                </button>
            </form>
        </div>
    </div>

    <script>
        // === CORE CHATBOT LOGIC ===
        
        // We now use a relative path since the front-end and back-end are served from the same origin.
        const serviceUrl = "/generate_text";

        const chatContainer = document.getElementById('chat-container');
        const chatForm = document.getElementById('chat-form');
        const messageInput = document.getElementById('message-input');

        // Function to create a message bubble
        function createMessageBubble(text, isUser) {
            const messageElement = document.createElement('div');
            messageElement.classList.add('flex', 'items-start', isUser ? 'justify-end' : 'justify-start');
            
            const textElement = document.createElement('div');
            textElement.classList.add(
                'rounded-2xl',
                'p-4',
                'max-w-[80%]',
                'shadow-md',
                isUser ? 'bg-blue-600' : 'bg-gray-200',
                isUser ? 'text-white' : 'text-gray-800'
            );
            textElement.textContent = text;
            messageElement.appendChild(textElement);
            chatContainer.appendChild(messageElement);
            
            // Scroll to the latest message
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Function to handle form submission
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const userMessage = messageInput.value.trim();
            if (!userMessage) return;

            createMessageBubble(userMessage, true);
            messageInput.value = '';

            try {
                const response = await fetch(serviceUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: userMessage }),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();
                
                if (data.response) {
                    createMessageBubble(data.response, false);
                } else {
                    createMessageBubble("Sorry, I didn't receive a valid response from the chatbot.", false);
                }

            } catch (error) {
                console.error('Fetch error:', error);
                createMessageBubble("Sorry, I encountered an error. Please try again.", false);
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def serve_index():
    """Serves the main HTML page for the chatbot."""
    return HTML_CONTENT

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
