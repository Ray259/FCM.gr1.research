import numpy as np

class FCM:
    def __init__(self, data, clusters=2, m=2, eps=0.01, lmax=50):
        self.c = clusters
        self.lmax = lmax
        self.m = m
        self.eps = eps
        self.Y = data
        self.u = np.zeros((self.c, self.Y.shape[0]))
        self.centers = np.zeros((self.c, self.Y.shape[1]))
        self.members = np.zeros((self.Y.shape[0]))  # points belong to which cluster
        self.cluster_members = np.zeros(self.c) # clusters's number of members

    def cal_centers(self):
        n_points, n_features = self.Y.shape   
        n_clusters = self.c
        centers = np.zeros((n_clusters, n_features))

        for i in range(n_clusters):
            for j in range(n_features):
                u_k_m_sum = 0
                u_k_m_yk_sum = 0
                for k in range(n_points):                
                    u_k_m = self.u[i, k] ** self.m
                    u_k_m_sum += u_k_m
                    u_k_m_yk_sum += u_k_m * self.Y[k, j]
                centers[i, j] = u_k_m_yk_sum / u_k_m_sum               
        self.centers = centers        
        return centers

    def update_membership(self, centers):
        n_points, n_features = self.Y.shape         
        n_clusters = self.c
        updated_u = np.zeros((n_clusters, n_points))
        for i in range(n_clusters):
            for k in range(n_points):
                d = 0
                d_ik = np.linalg.norm(self.Y[k] - centers[i]) 
                for j in range(n_clusters):  
                    d_jk = np.linalg.norm(self.Y[k] - centers[j])  
                    d += (d_ik / d_jk) ** (2 / (self.m - 1))
                updated_u[i, k] = d ** -1
        return updated_u
        
    def update_cluster_members(self):
        n_points, n_features = self.Y.shape
        u = self.u
        for k in range(n_points):
            t_cluster = int(np.argmax(u[:,k]))
            self.members[k] = t_cluster
            self.cluster_members[t_cluster] += 1            

    def loop(self):
        n_points, n_features = self.Y.shape
        n_clusters = self.c
        self.u = np.random.rand(n_clusters, n_points)
        self.u = self.u / np.sum(self.u, axis=0)
        for l in range(self.lmax):
            print(l, "/", self.lmax)
            centers = self.cal_centers()
            updated_u = self.update_membership(centers)
            if np.linalg.norm(updated_u - self.u) < self.eps:
                break
            self.u = updated_u
        print("Complete")
        self.data_center = np.mean(self.Y, axis=0)
        return updated_u

class sSFCM(FCM):
    def __init__(self, data, supervised_membership, clusters=2, m=2, eps=0.01, lmax=50, alpha=0.5):
        super().__init__(data, clusters, m, eps, lmax)
        self.u_supervised = supervised_membership
        self.alpha = alpha

    def update_membership(self, centers):
        updated_u = super().update_membership(centers)
        updated_u = (1 - self.alpha) * updated_u + self.alpha * self.u_supervised
        return updated_u

class eSFCM(sSFCM):
    def __init__(self, data, supervised_membership, clusters=2, m=2, eps=0.01, lmax=50, alpha=0.5, beta=1.0):
        super().__init__(data, supervised_membership, clusters, m, eps, lmax, alpha)
        self.beta = beta

    def update_membership(self, centers):
        updated_u = super().update_membership(centers)
        entropy_term = -np.sum(updated_u * np.log(updated_u + 1e-10), axis=0)
        updated_u += self.beta * entropy_term
        updated_u = updated_u / np.sum(updated_u, axis=0)
        return updated_u
