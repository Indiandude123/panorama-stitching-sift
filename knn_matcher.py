from sklearn.neighbors import NearestNeighbors

def custom_knn_matcher(descriptors1, descriptors2, k=2, ratio=0.75):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(descriptors2)
    distances, indices = nbrs.kneighbors(descriptors1)
    
    matches = []
    for i, neighbors in enumerate(indices):
        if len(neighbors) < 2:
            continue 
        if distances[i][0] < ratio * distances[i][1]:
            neighbor = neighbors[0]
            distance = distances[i][0]
            matches.append((i, neighbor, distance))
    
    return matches