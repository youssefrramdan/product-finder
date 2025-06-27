import streamlit as st
import pandas as pd
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load spacy model (you'll need to download it first using: python -m spacy download en_core_web_sm)
@st.cache_resource
def load_model():
    return spacy.load('en_core_web_sm')

nlp = load_model()

# Load product dataset
@st.cache_data
def load_data():
    df = pd.read_csv("DataSetProducts.csv")
    df["combined_text"] = df["name"].fillna('') + " " + df["description"].fillna('') + " " + df["categoryName"].fillna('')
    return df

df = load_data()

# Create embeddings using spacy
@st.cache_data
def create_embeddings(texts):
    return np.array([nlp(text).vector for text in texts])

product_embeddings = create_embeddings(df["combined_text"].tolist())

# Recommendation function based on user input
def recommend_for_custom_input(user_input, top_n=5):
    input_embedding = nlp(user_input).vector.reshape(1, -1)
    similarities = cosine_similarity(input_embedding, product_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    return df.iloc[top_indices][["name", "price", "categoryName", "imageUrls"]]

# Streamlit UI
st.title("🔍 Smart Product Finder")
user_input = st.text_input("Describe the product you're looking for:")

if user_input:
    st.subheader("🛍 Recommended Products:")
    results = recommend_for_custom_input(user_input)

    for _, row in results.iterrows():
        st.markdown(f"### {row['name']}")
        st.image(row["imageUrls"].strip("[]").split(",")[0].replace("'", "").strip(), width=200)
        st.write(f"💵 Price: {row['price']} EGP")
        st.write(f"📂 Category: {row['categoryName']}")
        st.markdown("---")
