import numpy as np
from typing import Dict, Any
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(true_labels: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    """Оценка кластеризации с F1-мерой"""
    unique_clusters = set(clusters)
    if len(unique_clusters) - (1 if -1 in unique_clusters else 0) < 2:
        return {'f1_score': 0.0, 'adjusted_rand': 0.0}

    return {
        'f1_score': f1_score(true_labels, clusters, average='weighted'),
        'adjusted_rand': adjusted_rand_score(true_labels, clusters)
    }

def evaluate_clustering_internal(X_embedded: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
    """Оценка кластеризации внутренними метриками"""
    if len(set(clusters)) > 1:
        return {
            'silhouette': silhouette_score(X_embedded, clusters),
            'calinski_harabasz': calinski_harabasz_score(X_embedded, clusters),
            'davies_bouldin': davies_bouldin_score(X_embedded, clusters),
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'noise_percent': 100 * (clusters == -1).mean()
        }
    return {
        'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
        'noise_percent': 100 * (clusters == -1).mean()
    }