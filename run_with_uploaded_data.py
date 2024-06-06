import streamlit as st
import numpy as np
import pandas as pd
from FCM import FCM, sSFCM, eSFCM
from criteria import Criteria

# Global variables to store data and results
data = None
u_supervised = None
algorithm_result = None

def load_data(uploaded_file):
    return np.loadtxt(uploaded_file, delimiter=',')

def run_fcm_algorithm(Y, fcm_type, clusters, m, eps, lmax, alpha=0.5, beta=1.0, u_supervised=None):
    if fcm_type == "Unsupervised FCM":
        algorithm = FCM(data=Y, clusters=clusters, m=m, eps=eps, lmax=lmax)
    elif fcm_type == "Semi-Supervised FCM":
        algorithm = sSFCM(data=Y, supervised_membership=u_supervised, clusters=clusters, m=m, eps=eps, lmax=lmax, alpha=alpha)
    elif fcm_type == "Entropy Regularized FCM":
        algorithm = eSFCM(data=Y, supervised_membership=u_supervised, clusters=clusters, m=m, eps=eps, lmax=lmax, alpha=alpha, beta=beta)

    updated_u = algorithm.loop()
    algorithm.update_cluster_members()
    data_point_cluster_map = {j: algorithm.members[j] for j in range(algorithm.Y.shape[0])}
    return algorithm, updated_u, data_point_cluster_map

def main():
    global data, u_supervised, algorithm_result

    st.title('FCM Algorithm Demo')

    st.sidebar.header('Input Data and Parameters')

    uploaded_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'txt'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.sidebar.write('Uploaded Data:')
        st.sidebar.write(data.shape[0], ' items.')
        st.sidebar.write(data.T)
    else:
        if data is None:
            data = np.zeros((100, 2))
        st.sidebar.write('Please upload your data.')

    fcm_type = st.sidebar.selectbox("FCM Type", ["Unsupervised FCM", "Semi-Supervised FCM", "Entropy Regularized FCM"])
    alpha = 0.5
    beta = 1.0
    if fcm_type != "Unsupervised FCM":
        supervised_labels = st.sidebar.file_uploader("Upload Supervised Membership File", type=['csv', 'txt'])
        if supervised_labels is not None:
            u_supervised = load_data(supervised_labels)
            alpha = st.sidebar.slider("Alpha (Weight for supervised information)", min_value=0.0, max_value=1.0, value=0.5)
        if fcm_type == "Entropy Regularized FCM":
            beta = st.sidebar.slider("Beta (Weight for entropy regularization)", min_value=0.0, max_value=5.0, value=1.0)

    clusters = st.sidebar.number_input('Number of Clusters', min_value=2, max_value=100, value=2)
    m = st.sidebar.number_input('Fuzziness Coefficient (m)', min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    eps = st.sidebar.number_input('Epsilon', min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    lmax = st.sidebar.number_input('Maximum number of iterations', min_value=10, max_value=100, value=50)

    st.header('Output')

    if st.button('Run Algorithm'):
        st.write('Running Algorithm...')
        algorithm_result = run_fcm_algorithm(data, fcm_type, clusters, m, eps, lmax, alpha, beta, u_supervised)
        st.write('Algorithm completed.')

    if algorithm_result is not None:
        o, updated_u, data_point_cluster_map = algorithm_result

        st.write('Final Membership:')
        st.write(updated_u)

        clusters_df = pd.DataFrame(o.centers, columns=[f"Feature {i+1}" for i in range(o.centers.shape[1])])
        clusters_df.index += 1
        clusters_df.index.name = "Cluster"
        clusters_df.rename(index=lambda x: f"Cluster {x}", inplace=True)
        st.write('Cluster Centers:')
        st.write(clusters_df)

        cr = Criteria(o.centers, o.cluster_members, o.members, o.Y.shape[0], clusters, o.data_center, o.Y, o.u, o.m)
        crt = cr.validate()
        for c in crt:
            st.write(c[0] + ':', c[1])

        datapoint_index = st.number_input('Enter data point index to see its cluster', min_value=0, max_value=o.Y.shape[0]-1, step=1)
        if st.button('Lookup Cluster'):
            cluster = data_point_cluster_map.get(datapoint_index, None)
            if cluster is not None:
                st.write(f'Data point {datapoint_index} belongs to cluster {cluster + 1}')
            else:
                st.write(f'Data point {datapoint_index} does not belong to any cluster.')

        with st.expander("Summary"):
            for i in range(clusters):
                indices = [index for index, cluster in data_point_cluster_map.items() if cluster == i]
                if indices:
                    points = [(j, o.Y[j]) for j in indices]
                    cluster_df = pd.DataFrame(points, columns=['Point Index', 'Features'])
                    st.write(f'Cluster {i+1} Points ({len(cluster_df)} data points):')
                    st.write(cluster_df)
                else:
                    st.write(f'Cluster {i+1} is empty.')

if __name__ == '__main__':
    main()

css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 400px;
        max-width: 800px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)
