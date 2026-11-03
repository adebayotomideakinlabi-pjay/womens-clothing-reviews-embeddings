# womens-clothing-reviews-embeddings
Women's Clothing E-Commerce Reviews: embedding-based theme &amp; sentiment analysis.
# Women's Clothing Reviews: Text Embeddings

Analysing customer reviews using OpenAI text embeddings.

## What I did
1. Generated text embeddings for the reviews with the OpenAI API
2. Reduced them to 2D with t-SNE and plotted them
3. Sorted reviews into topics (quality, fit, style, comfort) using cosine distance
4. Built a similarity search with ChromaDB to find the closest reviews

## Results
![t-SNE plot](tsne_plot.png)

Topic results are in the CSV files in this repository.

## Code
See `reviews_analysis.py`.

## Tools
Python, OpenAI API, scikit-learn, SciPy, ChromaDB, pandas, matplotlib

## Data
Women's Clothing E-Commerce Reviews dataset (provided through DataCamp).
