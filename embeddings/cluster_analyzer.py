from embeddings.nebula_embeddings_api import EmbeddingsLoader
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn import metrics

index = EmbeddingsLoader()
X = index.get_all_vectors()

for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, distance_threshold=6, n_clusters=None, compute_full_tree=True)
    clustering.fit(X)
    print(linkage)
    print(clustering.n_clusters_ )
    print(clustering.labels_, len(clustering.labels_))
    #print("Number of clusters: ", np.unique(clustering.labels_).size)

    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, clustering.labels_, metric='sqeuclidean'))
        