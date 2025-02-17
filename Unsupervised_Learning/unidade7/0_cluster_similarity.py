import numpy as np

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compute_single_linkage(cluster1, cluster2):
    return min(euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2)

def compute_complete_linkage(cluster1, cluster2):
    return max(euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2)

def compute_average_linkage(cluster1, cluster2):
    distances = [euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2]
    return np.mean(distances)

def compute_centroid_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return euclidean_distance(centroid1, centroid2)

cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("Similaridade ligação simples: ", compute_single_linkage(cluster1, cluster2))
print("Similaridade ligação completa: ", compute_complete_linkage(cluster1, cluster2))
print("Similaridade ligação média: ", compute_average_linkage(cluster1, cluster2))
print("Similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1, cluster2))
