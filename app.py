from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pre-trained sentence embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to calculate similarity score between two sentences
def calculate_similarity(sentence1, sentence2):
    # Tokenize and encode sentences
    embeddings = model.encode([sentence1, sentence2])
    # Calculate cosine similarity between sentence embeddings
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # Normalize similarity score to range [0, 1]
    normalized_similarity_score = (similarity_score + 1) / 2
    return normalized_similarity_score

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_api():
    # Get request data
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    # Calculate similarity score
    similarity_score = calculate_similarity(text1, text2)

    # Prepare response
    response = {'similarity score': similarity_score}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
