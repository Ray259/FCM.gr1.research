import numpy as np

# Calinski-Harabasz index
def VRC(centers, n_cluster_members, members, n_points, n_clusters, data_center, data):    
    WGSSj = np.zeros(n_clusters)
    WGSS = 0
    BGSS = 0

    for j in range(n_points):  
        t = int(members[j])
        WGSSj[t] += np.linalg.norm(data[j] - centers[t]) ** 2

    for i in range(n_clusters):
        BGSS += n_cluster_members[i] * np.linalg.norm(centers[i] - data_center) ** 2
        WGSS += WGSSj[i]        
    return  (BGSS * (n_points - n_clusters)) / (WGSS * (n_clusters - 1))