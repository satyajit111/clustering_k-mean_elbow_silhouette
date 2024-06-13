from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
def text_to_cluster(texts):

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorized_data = vectorizer.fit_transform(texts)
    k=5
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(vectorized_data)
    cluster_labels = kmeans.labels_.tolist()
    cluster_centroids = kmeans.cluster_centers_.tolist()

    return cluster_labels, cluster_centroids




#data= pd.read_csv()
data = pd.read_csv(r"C:\Users\baps\Desktop\VEROFAX\product.csv")
data = data.dropna(subset=['name'])

data_column = data["name"].tolist()
print(data_column)

cl,cc = text_to_cluster(data_column)


print("Cluster LABEL",cl)
print("Cluster centroid",cc)

data['cluster_label'] = cl

data.to_csv(r"C:\Users\baps\Desktop\VEROFAX\product_with_clusters_new.csv", index=False)