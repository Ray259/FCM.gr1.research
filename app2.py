from flask import Flask, request, jsonify
import numpy as np
from FCM import FCM, sSFCM, eSFCM
from criteria import Criteria

app = Flask(__name__)

@app.route('/run_fcm', methods=['POST'])
def run_fcm():
    data = request.json['data']
    fcm_type = request.json['fcm_type']
    clusters = request.json['clusters']
    m = request.json['m']
    eps = request.json['eps']
    lmax = request.json['lmax']
    alpha = request.json.get('alpha', 0.5)
    beta = request.json.get('beta', 1.0)
    u_supervised = request.json.get('u_supervised', None)

    Y = np.array(data)
    u_supervised = np.array(u_supervised) if u_supervised is not None else None

    if fcm_type == "Unsupervised FCM":
        algorithm = FCM(data=Y, clusters=clusters, m=m, eps=eps, lmax=lmax)
    elif fcm_type == "Semi-Supervised FCM":
        algorithm = sSFCM(data=Y, supervised_membership=u_supervised, clusters=clusters, m=m, eps=eps, lmax=lmax, alpha=alpha)
    elif fcm_type == "Entropy Regularized FCM":
        algorithm = eSFCM(data=Y, supervised_membership=u_supervised, clusters=clusters, m=m, eps=eps, lmax=lmax, alpha=alpha, beta=beta)

    updated_u = algorithm.loop()
    algorithm.update_cluster_members()
    data_point_cluster_map = {j: int(algorithm.members[j]) for j in range(algorithm.Y.shape[0])}

    centers = algorithm.centers.tolist()
    updated_u = updated_u.tolist()

    return jsonify({
        'centers': centers,
        'updated_u': updated_u,
        'data_point_cluster_map': data_point_cluster_map
    })

if __name__ == '__main__':
    app.run(debug=True)
