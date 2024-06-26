import numpy as np

class Criteria():
    def __init__(self, centers, n_cluster_members, members, n_points, n_clusters, data_center, data, u, m):
        self.centers = centers
        self.n_cluster_members = n_cluster_members
        self.members = members
        self.n_points = n_points
        self.n_clusters = n_clusters
        self.data_center = data_center
        self.data = data
        self.u = u
        self.m = m
    
    # Partition coefficient
    def partition_coefficient(self):        
        u = self.u
        pc = 0
        for k in range(self.n_points):
            for i in range(self.n_clusters):
                pc += u[i, k] ** 2
        pc = pc / self.n_points
        return ('Partition Coefficient', pc)
        
    # Entropy
    def entropy(self, a = 2):
        u = self.u
        e = 0
        for k in range(self.n_points):
            for i in range(self.n_clusters):
                e += u[i, k] * np.log(u[i, k]) / np.log(a)
        e = -e / self.n_points
        return ('Entropy', e)
    
    # Calinski-Harabasz index
    def VRC(self):
        WGSSj = np.zeros(self.n_clusters)
        WGSS = 0
        BGSS = 0

        for j in range(self.n_points):
            t = int(self.members[j])  # number of points in cluster that point j belongs to
            WGSSj[t] += np.linalg.norm(self.data[j] - self.centers[t]) ** 2

        for i in range(self.n_clusters):
            BGSS += self.n_cluster_members[i] * np.linalg.norm(self.centers[i] - self.data_center) ** 2
            WGSS += WGSSj[i]
        
        VRC = (BGSS * (self.n_points - self.n_clusters)) / (WGSS * (self.n_clusters - 1))
        return ('Calinski-Harabasz index', VRC)

    # Davies–Bouldin index
    def DBI(self):
        dlm = np.zeros((self.n_clusters, self.n_clusters))  # between cluster
        dl = np.zeros(self.n_clusters)  # within cluster
        Dlm = np.zeros((self.n_clusters, self.n_clusters))
        DBI = 0
        for j in range(self.n_points):
            t = int(self.members[j])  # number of points in cluster that point j belongs to
            dl[t] += np.linalg.norm(self.data[j] - self.centers[t]) ** 2 / self.n_cluster_members[t]

        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if (i != j):
                    dlm[i, j] = np.linalg.norm(self.centers[i] - self.centers[j])
                    Dlm[i, j] = (dl[i] + dl[j]) / dlm[i, j]
            Dl = np.max(Dlm[i])
            DBI += Dl

        DBI = DBI / self.n_clusters
        return ('Davies–Bouldin index', DBI)
    
    
    # Dunn index
    def DNI(self):
        DNI = 0
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                if (i != j):
                    DNI += np.linalg.norm(self.centers[i] - self.centers[j])
        return ('Dunn index', DNI)
    
    
    
    def validate(self):
        vrc = self.VRC()
        dbi = self.DBI()
        pc = self.partition_coefficient()
        e = self.entropy()
        
        return [pc, e, vrc, dbi]