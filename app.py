import flask
import os
import google.generativeai as genai
import numpy as np
import sklearn.metrics.pairwise as pairwise
import time

# Get the API key from the environment variable.
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("The 'API_KEY' environment variable is not set.")

genai.configure(api_key=API_KEY)

# Use the latest model
model = genai.GenerativeModel("gemini-1.5-flash-latest")
document = ""
document_embeddings = []

def load_and_process_document():
    """
    Loads the document, splits it into chunks, and generates embeddings for each chunk.
    """
    global document
    global document_embeddings
    
    document_file_path = "PSN_Finance_Document.txt"
    
    try:
        with open(document_file_path, 'r', encoding='utf-8') as f:
            document = f.read()
    except FileNotFoundError:
        print(f"Error: Document file '{document_file_path}' not found.")
        return False
        
    document_chunks = [chunk.strip() for chunk in document.split('\n\n') if chunk.strip()]
    document_embeddings = []
    
    for chunk in document_chunks:
        try:
            response = genai.embed_content(model="models/embedding-001", content=chunk)
            document_embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error generating embedding for a chunk: {e}")
            return False
            
    document_embeddings = np.array(document_embeddings)
    return document_chunks, True

def find_most_relevant_chunk(user_prompt_embedding):
    """
    Finds the most relevant document chunk based on the user's prompt.
    """
    similarities = pairwise.cosine_similarity(user_prompt_embedding.reshape(1, -1), document_embeddings)
    most_relevant_chunk_index = np.argmax(similarities)
    return most_relevant_chunk_index

def generate_response(user_prompt):
    """
    Generates a response using the Gemini model.
    """
    try:
        # Load the document and chunks here, just before generating a response.
        document_chunks, success = load_and_process_document()
        if not success:
            return "Error: The document could not be loaded. Please check the backend setup."

        # Get the embedding for the user prompt.
        user_prompt_embedding = genai.embed_content(model="models/embedding-001", content=user_prompt)
        most_relevant_chunk_index = find_most_relevant_chunk(user_prompt_embedding['embedding'])
        relevant_chunk = document_chunks[most_relevant_chunk_index]

        # Instruct the model to answer based ONLY on the relevant chunk.
        system_prompt = (
            "You are an expert on the provided document."
            "Answer the user's question ONLY based on the following context."
            "If the answer is not in the context, state that you cannot find the information in the provided document."
            "Context: " + relevant_chunk
        )

        final_prompt = system_prompt + "\n\nUser Question: " + user_prompt
        
        response = model.generate_content(final_prompt)
        return response.text
        
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "An error occurred while generating the response. Please try again."

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    """
    Renders the main page.
    """
    return flask.render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles chat requests.
    """
    user_prompt = flask.request.json.get("prompt")
    if not user_prompt:
        return flask.jsonify({"response": "Please provide a prompt."})

    response_text = generate_response(user_prompt)
    
    return flask.jsonify({"response": response_text})

if __name__ == '__main__':
    # When running locally, use this. On Cloud Run, Gunicorn will handle it.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

