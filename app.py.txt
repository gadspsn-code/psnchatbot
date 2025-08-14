import os
import flask
import google.generativeai as genai
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = flask.Flask(__name__)

# Configure the Gemini API key from an environment variable for security
api_key = os.environ.get("API_KEY")
if not api_key:
    # This will stop the app from running if the key isn't set, which is a good safety check.
    raise ValueError("The 'API_KEY' environment variable is not set.")

genai.configure(api_key=api_key)

# Global variables to store document data
document_chunks = []
document_embeddings = []
text_model = genai.GenerativeModel('gemini-1.5-flash')

def clean_text(text):
    """Simple function to clean up text for better processing."""
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=2500):
    """Breaks a large text into smaller, manageable chunks."""
    chunks = []
    current_chunk = ""
    words = text.split()
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_and_process_document(file_path):
    """Loads and processes the document when the app starts."""
    global document_chunks, document_embeddings
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
        
        document_text = clean_text(document_text)
        document_chunks = chunk_text(document_text)

        document_embeddings_response = genai.embed_content(
            model="models/embedding-001",
            content=document_chunks
        )
        document_embeddings = np.array(document_embeddings_response['embedding'])
        print("Document loaded and processed successfully!")
    except FileNotFoundError:
        print(f"Error: Document file '{file_path}' not found.")
        document_chunks = []
        document_embeddings = []
    except Exception as e:
        print(f"An error occurred during document processing: {str(e)}")
        document_chunks = []
        document_embeddings = []

@app.route("/")
def index():
    """Serves the front-end HTML file."""
    return flask.render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests and provides context-aware responses."""
    global document_chunks, document_embeddings
    
    if not document_chunks:
        return flask.jsonify({"error": "The document could not be loaded. Please check the backend setup."}), 500

    data = flask.request.json
    user_prompt = data.get("prompt")
    if not user_prompt:
        return flask.jsonify({"error": "No prompt provided"}), 400

    try:
        # Create an embedding for the user's question
        prompt_embedding_response = genai.embed_content(
            model="models/embedding-001",
            content=user_prompt
        )
        prompt_embedding = np.array(prompt_embedding_response['embedding'])

        # Find the most relevant chunks from the document
        similarities = cosine_similarity(prompt_embedding.reshape(1, -1), document_embeddings)[0]
        most_relevant_chunk_index = np.argmax(similarities)
        relevant_chunk = document_chunks[most_relevant_chunk_index]

        # Instruct the model to answer based ONLY on the relevant chunk
        system_prompt = (
            "You are an expert on the provided document. "
            "Answer the user's question ONLY based on the following context. "
            "If the answer is not in the context, state that you cannot find the information in the provided document. "
            "Context: " + relevant_chunk
        )
        
        final_prompt = system_prompt + "\n\nUser Question: " + user_prompt

        response = text_model.generate_content(final_prompt)
        return flask.jsonify({"response": response.text})
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    document_file_path = "PSN_Finance_Document.txt"
    load_and_process_document(document_file_path)

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
