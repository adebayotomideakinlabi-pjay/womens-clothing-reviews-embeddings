"""
Women's Clothing E-Commerce Reviews: embedding-based theme & sentiment analysis.

Usage
-----
  # Real embeddings (needs OPENAI_API_KEY and network access)
  python review_embedding_analysis.py reviews.csv --backend openai

  # Offline fallback (TF-IDF + SVD), useful for testing without an API
  python review_embedding_analysis.py reviews.csv --backend tfidf

Outputs (in ./output): clusters.csv, cluster_summary.csv, cluster_map.png,
rating_by_cluster.png, plus a printed search demo.
"""
# pip install openai chromadb scikit-learn scipy pandas matplotlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from scipy.spatial import distance
from sklearn.manifold import TSNE

try:
    from IPython.display import display   # renders DataFrames as tables in Jupyter/Colab
except ImportError:
    display = print

pd.set_option("display.max_colwidth", 120)   # show more of each review in tables

EMBED_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- Load the reviews ----------
df = pd.read_csv("womens_clothing_e-commerce_reviews.csv")
reviews = df["Review Text"].dropna().astype(str).str.strip().tolist()
# reviews = reviews[:1000]   # uncomment for a faster, cheaper test run
print(f"{len(reviews)} reviews")

# ---------- 1. Generating the embeddings ----------
def embed(texts, batch_size=500):
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(input=texts[i:i + batch_size], model=EMBED_MODEL)
        vectors.extend(item.embedding for item in response.data)
    return vectors

embeddings = embed(reviews)          # list of vectors

# ---------- 2. Reducing the dimensionality and visualizing ----------
tsne = TSNE(n_components=2, perplexity=min(30, len(reviews) - 1),
            init="pca", random_state=42)
embeddings_2d = tsne.fit_transform(np.array(embeddings))   # numpy array, shape (n, 2)

plt.figure(figsize=(9, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6, alpha=0.5)
plt.title("Customer reviews in 2D (t-SNE of text embeddings)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.show()

# ---------- 3. Categorizing the reviews into topics ----------
topics = ["quality", "fit", "style", "comfort"]
topic_embeddings = embed(topics)

# cosine DISTANCE: smaller = closer match
distances = np.array([[distance.cosine(r, t) for t in topic_embeddings]
                      for r in embeddings])          # shape (n_reviews, n_topics)

# Table 1: the 3 closest reviews for each topic
rows = []
for j, topic in enumerate(topics):
    for i in np.argsort(distances[:, j])[:3]:
        rows.append({"topic": topic,
                     "cosine_distance": round(distances[i, j], 3),
                     "review": reviews[i]})
top_reviews_table = pd.DataFrame(rows)
display(top_reviews_table)

# Best-matching topic for every review
df_topics = pd.DataFrame({"review": reviews,
                          "topic": [topics[k] for k in distances.argmin(axis=1)]})

# Table 2: number of reviews per topic
topic_counts = (df_topics["topic"].value_counts()
                .rename_axis("topic").reset_index(name="n_reviews"))
display(topic_counts)

# ---------- 4. Finding similar reviews (ChromaDB) ----------
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL)

chroma_client = chromadb.Client()

# FIX: Use get_or_create_collection to avoid UniqueConstraintError
collection = chroma_client.get_or_create_collection(
    name="reviews",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},   # default is L2; cosine matches step 3
)

ids = [str(i) for i in range(len(reviews))]
for i in range(0, len(reviews), 500):    # add in chunks to respect batch limits
    collection.add(documents=reviews[i:i + 500], ids=ids[i:i + 500])

def find_closest_reviews(input_review, n=3):
    results = collection.query(query_texts=[input_review], n_results=n + 1)
    docs = results["documents"][0]
    return [d for d in docs if d != input_review][:n]   # drop the review itself

most_similar_reviews = find_closest_reviews(reviews[0])   # plain list, as the project requires

# Table 3: display copy of most_similar_reviews
similar_table = pd.DataFrame({"rank": [1, 2, 3], "review": most_similar_reviews})
display(similar_table)
