from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import json

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://1c567b57-a9a3-4959-95a8-a20ce9b0f0d5.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="ICfCrvnuKA10Swwy7giPRy3P6uQxKDn8dgefeNYpfIuqqV_j4ifE4w",
)

# Initialize your embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Your similarity threshold and collection name
collection_name = "planets"
similarity_threshold = 0.5

# Initialize Groq client
groq_client = Groq(api_key="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")

# Define the function that processes the query
def query_planet(query_text):
    # Encode the query into a vector
    query_vector = model.encode(query_text).tolist()

    # Search for the most similar vectors
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1  # Return the top match
    )

    if search_results and 'vector' in search_results[0]:
        closest_match = search_results[0]
        closest_vector = closest_match['vector']

        # Calculate the cosine similarity
        query_vector_reshaped = np.array(query_vector).reshape(1, -1)
        closest_vector_reshaped = np.array(closest_vector).reshape(1, -1)

        similarity = cosine_similarity(query_vector_reshaped, closest_vector_reshaped)[0][0]

        if similarity >= similarity_threshold:
            return closest_match['payload']
        else:
            return "This query is not related to the solar system."
    else:
        return "Please enter a query about the solar system."

# Define the function that interacts with Groq
def generate_chat_response(user_query, context_response):
    combined_content = f"Context: {context_response[:100]}...\nUser's query: {user_query}"

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant for solar system and planets system."},
            {"role": "user", "content": combined_content},
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content

# Define the API route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Log request size
    print(f"Request size: {len(json.dumps(data))} bytes")

    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = data['query']

    # Get the response from Qdrant
    context_response = query_planet(user_query)

    # Generate chat response using Groq
    chat_response = generate_chat_response(user_query, context_response)

    # Return the response as JSON
    return jsonify({'response': chat_response})

if __name__ == '__main__':
    app.run(debug=True)
