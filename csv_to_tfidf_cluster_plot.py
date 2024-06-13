from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1)
    sse = []
    silhouette_scores = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Cluster Centers')
    plt.xticks(iters)
    plt.ylabel('SSE')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(iters, silhouette_scores, marker='o')
    plt.xlabel('Cluster Centers')
    plt.xticks(iters)
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores')

    plt.show()

def text_to_cluster(texts, n_clusters):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorized_data = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(vectorized_data)

    cluster_labels = kmeans.labels_
    return cluster_labels, vectorized_data, kmeans.cluster_centers_

# Load data
data = pd.read_csv(r"C:\Users\baps\Desktop\VEROFAX\product.csv")
data = data.dropna(subset=['name'])
data_column = data["name"].tolist()

# Vectorize data
vectorizer = TfidfVectorizer(stop_words="english")
vectorized_data = vectorizer.fit_transform(data_column)

# Find optimal number of clusters
find_optimal_clusters(vectorized_data, max_k=10)

# Assuming after visual inspection, let's say the optimal number of clusters is 3
optimal_clusters = 3
cluster_labels, vectorized_data, cluster_centroids = text_to_cluster(data_column, optimal_clusters)

# Print cluster labels
print("Cluster Labels:")
for text, label in zip(data_column, cluster_labels):
    print(f"Text: {text} - Cluster: {label}")

# Check if related rows fall in proper clusters using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

print("\nChecking related rows using cosine similarity:")
for i, text_vector in enumerate(vectorized_data):
    similarities = cosine_similarity(text_vector, vectorized_data).flatten()
    similar_indices = similarities.argsort()[-2:-7:-1]  # Get top 5 similar indices excluding the current text
    #print(f"\nText: {data_column[i]} - Cluster: {cluster_labels[i]}")
    #print("Most similar texts:")
    for idx in similar_indices:
        print(f"    {data_column[idx]} - Cluster: {cluster_labels[idx]} - Similarity: {similarities[idx]:.2f}")

# Printing cluster centroids (optional, just to see the values)
print("\nCluster Centroids:")
print(cluster_centroids)
