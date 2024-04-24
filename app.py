import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from FCM import FCM
from criteria import *

st.set_page_config(layout='wide')
def main():
    st.title('FCM Algorithm Demo')

    Y = np.zeros((100, 2))
    
    

    st.sidebar.header('Input Data and Parameters')

    data_generated = st.sidebar.checkbox('Generate Random Data')
    if data_generated:
        n_samples = st.sidebar.number_input('Number of samples', value=100)
        Y = np.random.randint(100, size=(n_samples, 2))
        st.sidebar.write('Generated Data:')
        st.sidebar.write(Y.T)
    else:
        st.sidebar.write('Please upload your data.')

    clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=100, value=2)
    m = st.sidebar.slider('Fuzziness Coefficient (m)', min_value=1.1, max_value=5.0, value=2.0)
    eps = st.sidebar.slider('Epsilon', min_value=0.001, max_value=0.1, value=0.01)
    lmax = st.sidebar.slider('Maximum number of iterations', min_value=10, max_value=100, value=50)

    st.header('Output')

    o = FCM(data=Y, clusters=clusters, m=m, eps=eps, lmax=lmax)
    # st.write('Initial Membership:')
    # st.write(o.u)

    if st.button('Run Algorithm'):
        updated_u = o.loop()
        st.write('Final Membership:')
        st.write(updated_u)

        # Plot clusters
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Y[:, 0], Y[:, 1], c=np.argmax(updated_u, axis=0), cmap='viridis')
        ax.scatter(o.centers[:, 0], o.centers[:, 1], marker='X', s=100, c='red')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('FCM Clustering')
        ax.legend(['Data Points', 'Cluster Centers'], loc='upper right')
        st.pyplot(fig)
        
        # Criteria
        o.update_cluster_members()
        vrc = VRC(o.centers, o.cuslter_members, o.members, o.Y.shape[0], clusters, o.data_center, o.Y)
        st.write('VRC:', vrc)
        
if __name__ == '__main__':
    main()
