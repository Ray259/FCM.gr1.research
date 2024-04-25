import streamlit as st
import numpy as np
import pandas as pd
from FCM import FCM
from criteria import *

def main():
    st.title('FCM Algorithm Demo')

    Y = np.zeros((100, 2))
    
    st.sidebar.header('Input Data and Parameters')

    uploaded_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'txt'])
    if uploaded_file is not None:
        # Read the uploaded file
        Y = np.loadtxt(uploaded_file, delimiter=',')
        st.sidebar.write('Uploaded Data:')
        st.sidebar.write(Y.T)
    else:
        st.sidebar.write('Please upload your data.')

    clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=100, value=2)
    m = st.sidebar.slider('Fuzziness Coefficient (m)', min_value=1.1, max_value=5.0, value=2.0)
    eps = st.sidebar.slider('Epsilon', min_value=0.001, max_value=0.1, value=0.01)
    lmax = st.sidebar.slider('Maximum number of iterations', min_value=10, max_value=100, value=50)

    st.header('Output')

    o = FCM(data=Y, clusters=clusters, m=m, eps=eps, lmax=lmax)

    if st.button('Run Algorithm'):
        updated_u = o.loop()
        st.write('Final Membership:')
        st.write(updated_u)

        # Clusters
        clusters_df = pd.DataFrame(o.centers, columns=[f"Feature {i+1}" for i in range(o.centers.shape[1])])
        clusters_df.index += 1  # Bắt đầu index từ 1
        clusters_df.index.name = "Cluster"
        clusters_df.rename(index=lambda x: f"Cluster {x}", inplace=True)  # Đặt nhãn cluster
        st.write('Cluster Centers:')
        st.write(clusters_df)
        
        # Criteria
        o.update_cluster_members()
        vrc = VRC(o.centers, o.cluster_members, o.members, o.Y.shape[0], clusters, o.data_center, o.Y)
        st.write('VRC:', vrc)

if __name__ == '__main__':
    main()
