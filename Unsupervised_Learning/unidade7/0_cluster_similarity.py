import numpy as np
from scipy.cluster.hierarchy import centroid


def compute_single_linkage(cluster1,cluster2):
    distMin = np.inf
    for p1 in cluster1:
        for p2 in cluster2:
            distancia = np.linalg.norm(np.array(p1)-np.array(p2))
            if distancia < distMin:
                distMin = distancia

    return distMin

def compute_complete_linkage(cluster1, cluster2):
    distMax = 0
    for p1 in cluster1:
        for p2 in cluster2:
            distancia = np.linalg.norm(np.array(p1) - np.array(p2))
            if distancia > distMax:
                distMax = distancia

    return distMax

def compute_average_linkage(cluster1, cluster2):
    distancia = 0
    for p1 in cluster1:
        for p2 in cluster2:
            distancia += np.linalg.norm(np.array(p1) - np.array(p2))

    return distancia/(len(cluster1)*len(cluster2))

def compute_centroid_linkage(cluster1,cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return np.linalg.norm(centroid1-centroid2)


cluster1 = [[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]]
cluster2 = [[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1,cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1,cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1,cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1,cluster2))



