import streamlit as st
import numpy as np
import pandas as pd
from FCM import FCM
from criteria import Criteria

def main():
    st.title('FCM Algorithm Demo')

    Y = np.zeros((100, 2))
    cluster_indices = {}  # Define cluster_indices here
    
    st.sidebar.header('Input Data and Parameters')

    uploaded_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'txt'])
    if uploaded_file is not None:
        # Read the uploaded file
        Y = np.loadtxt(uploaded_file, delimiter=',')
        st.sidebar.write('Uploaded Data:')
        st.sidebar.write(Y.shape[0], ' items.')
        st.sidebar.write(Y.T)
    else:
        st.sidebar.write('Please upload your data.')

    clusters = st.sidebar.number_input('Number of Clusters', min_value=2, max_value=100, value=2)
    m = st.sidebar.number_input('Fuzziness Coefficient (m)', min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    eps = st.sidebar.number_input('Epsilon', min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    lmax = st.sidebar.number_input('Maximum number of iterations', min_value=10, max_value=100, value=50)

    st.header('Output')

    o = None

    if 'algorithm_run' not in st.session_state:
        st.session_state.algorithm_run = False
        st.session_state.updated_u = None
        st.session_state.o = None

    if st.button('Run Algorithm'):
        st.write('Running Algorithm...')
        o = FCM(data=Y, clusters=clusters, m=m, eps=eps, lmax=lmax)
        st.session_state.algorithm_run = True
        st.session_state.updated_u = o.loop()
        o.update_cluster_members()
        st.session_state.o = o
        st.write('Algorithm completed.')

        # Populate cluster_indices after running the algorithm
        for i in range(clusters):
            cluster_indices[i] = [j for j in range(o.Y.shape[0]) if o.members[j] == i]

    if st.session_state.algorithm_run:
        o = st.session_state.o
        updated_u = st.session_state.updated_u
        st.write('Final Membership:')
        st.write(updated_u)

        # Clusters
        clusters_df = pd.DataFrame(o.centers, columns=[f"Feature {i+1}" for i in range(o.centers.shape[1])])
        clusters_df.index += 1  
        clusters_df.index.name = "Cluster"
        clusters_df.rename(index=lambda x: f"Cluster {x}", inplace=True)
        st.write('Cluster Centers:')
        st.write(clusters_df)
        
        # Validity
        cr = Criteria(o.centers, o.cluster_members, o.members, o.Y.shape[0], clusters, o.data_center, o.Y, o.u, o.m)
        crt = cr.validate()
        for c in crt:
            st.write(c[0] + ':', c[1])

        # Display array of points belonging to each cluster
        if st.session_state.algorithm_run:
            with st.expander("Points Belonging to Each Cluster"):
                for i in range(clusters):
                    if i in cluster_indices:
                        st.write(f'Cluster {i+1} Points:')
                        indices = cluster_indices[i]
                        points = [(j, o.Y[j]) for j in indices]  # Include point indices
                        cluster_df = pd.DataFrame(points, columns=['Point Index', 'Features'])
                        st.write(cluster_df)
                    else:
                        st.write(f'Cluster {i+1} is empty.')

        # Data point cluster lookup
        # if o is not None:
        #     data_point_cluster_map = {}  # Precompute data point to cluster mapping
        #     for cluster, indices in cluster_indices.items():
        #         for index in indices:
        #             data_point_cluster_map[index] = cluster

        #     datapoint_index = st.number_input('Enter data point index to see its cluster', min_value=0, max_value=o.Y.shape[0]-1, step=1)
        #     if st.button('Lookup Cluster'):
        #         if datapoint_index in data_point_cluster_map:
        #             cluster = data_point_cluster_map[datapoint_index]
        #             st.write(f'Data point {datapoint_index} belongs to cluster {cluster + 1}')
        #         else:
        #             st.write(f'Data point {datapoint_index} does not belong to any cluster.')


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


        # if st.button('Generate t-SNE Visualization'):
        #     # Apply t-SNE for visualization
        #     perplexity = min(30, len(Y) - 1)
        #     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        #     Y_tsne = tsne.fit_transform(Y)

        #     # Plotting
        #     plt.figure(figsize=(10, 7))
        #     for i in range(clusters):
        #         indices = cluster_indices[i]
        #         plt.scatter(Y_tsne[indices, 0], Y_tsne[indices, 1], label=f'Cluster {i+1}', alpha=0.7)

        #     plt.title('t-SNE visualization of Clusters')
        #     plt.legend()
        #     st.pyplot(plt)

        # Expandable section for cluster members