from flask import Flask, request, jsonify
import pandas as pd
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load product dataset
df = pd.read_csv("DataSetProducts.csv")
df["combined_text"] = df["name"].fillna('') + " " + df["description"].fillna('') + " " + df["categoryName"].fillna('')

# Create embeddings using spacy
product_embeddings = np.array([nlp(text).vector for text in df["combined_text"].tolist()])

def recommend_for_custom_input(user_input, top_n=5):
    input_embedding = nlp(user_input).vector.reshape(1, -1)
    similarities = cosine_similarity(input_embedding, product_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    return df.iloc[top_indices][["name", "price", "categoryName", "imageUrls"]]

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Service is running"
    })

@app.route('/search', methods=['POST'])
def search_products():
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400

        query = data['query']
        top_n = data.get('top_n', 5)  # Optional parameter, defaults to 5

        results = recommend_for_custom_input(query, top_n)

        # Convert results to list of dictionaries
        products = []
        for _, row in results.iterrows():
            # Clean up the imageUrls string and convert to list
            image_urls = row["imageUrls"].strip("[]").split(",")
            image_urls = [url.strip().strip("'") for url in image_urls]

            products.append({
                "name": row["name"],
                "price": row["price"],
                "category": row["categoryName"],
                "imageUrls": image_urls
            })

        return jsonify({
            "results": products
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
