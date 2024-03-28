import numpy as np

class FCM:
    def __init__(self, data, clusters=2, m=2, eps=0.01, lmax=50):
        self.c = clusters
        self.lmax = lmax
        self.m = m
        self.eps = eps
        self.Y = data

    def cal_centers(self):
        n_points, n_features = self.Y.shape   
        n_clusters = self.c
        centers = np.zeros((self.c, n_features))

        for i in range(n_clusters):
            for j in range(n_features):
                u_k_m_sum = 0
                u_k_m_yk_sum = 0
                for k in range(n_points):                
                    u_k_m = self.u[i, k] ** self.m
                    u_k_m_sum += u_k_m
                    u_k_m_yk_sum += u_k_m * self.Y[k, j]
                centers[i, j] = u_k_m_yk_sum / u_k_m_sum               

        return centers
    
    def update_membership(self, centers):
        n_points, n_features = self.Y.shape         
        n_clusters = self.c
        updated_u = np.zeros((n_clusters, n_points))
        for i in range(n_clusters):
            for k in range(n_points):
                d = 0
                d_ik = np.linalg.norm(self.Y[k] - centers[i]) 
                for j in range(n_features):           
                    d_jk = np.linalg.norm(self.Y[k] - centers[j])  
                    d += (d_ik / d_jk) ** (2 / (self.m - 1))
                updated_u[i, k] = d ** -1
        return updated_u
        
    def loop(self):
        n_points, n_features = self.Y.shape
        n_clusters = self.c
        self.u = np.random.rand(n_clusters, n_points)
        self.u = self.u / np.sum(self.u, axis=0)
        for l in range(self.lmax):
            print(l, "/", self.lmax)
            centers = self.cal_centers()
            updated_u = self.update_membership(centers)
            print(updated_u)
            if np.linalg.norm(updated_u - self.u) < self.eps:
                break
            self.u = updated_u
        return updated_u
    




# Example
# Y = np.array([[1, 3], [2, 5], [4, 8], [7, 9]])
# u = np.array([[0.8, 0.7, 0.2, 0.1], 
#               [0.1, 0.1, 0.7, 0.8],
#               [0.1, 0.2, 0.1, 0.1],])
Y = np.random.randint(100, size=(10, 2))
print("Data: ", Y)
u = np.random.rand(2, 10)
u = u / np.sum(u, axis=0, keepdims=True)
m = 2

o = FCM(data=Y)
o.u = u

centers = o.cal_centers()
print("Centers:")
print(centers)

updated_u = o.update_membership(centers)
print("Updated membership:")
print(updated_u)

print("=====================================")
final = o.loop()