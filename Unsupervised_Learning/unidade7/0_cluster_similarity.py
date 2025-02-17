import numpy as np


def compute_single_linkage(cluster1, cluster2):
    minimum_distance = np.inf
    for p1 in cluster1:
        for p2 in cluster2:
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if distance < minimum_distance:
                minimum_distance = distance

    return minimum_distance


def compute_complete_linkage(cluster1, cluster2):
    maximum_distance = 0
    for p1 in cluster1:
        for p2 in cluster2:
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if distance > maximum_distance:
                maximum_distance = distance

    return maximum_distance


def compute_average_linkage(cluster1, cluster2):
    mean_distance = 0
    for p1 in cluster1:
        distances = []
        for p2 in cluster2:
            distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
        mean_distance += np.mean(distances)
    mean_distance /= len(cluster1)

    return mean_distance


def compute_centroid_linkage(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

    return np.linalg.norm(centroid1 - centroid2)


cluster1 = [[9.0, 8.0], [6.0, 4.0], [2.0, 10.0], [3.0, 6.0], [1.0, 0.0]]
cluster2 = [[7.0, 4.0], [1.0, 10.0], [6.0, 10.0], [1.0, 6.0], [7.0, 1.0]]

print("similaridade ligação simples: ", compute_single_linkage(cluster1, cluster2))
print("similaridade ligação completa: ", compute_complete_linkage(cluster1, cluster2))
print("similaridade ligação média: ", compute_average_linkage(cluster1, cluster2))
print("similaridade pelo método do centroide: ", compute_centroid_linkage(cluster1, cluster2))
