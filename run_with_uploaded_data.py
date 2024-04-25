import streamlit as st
import numpy as np
import pandas as pd
from FCM import FCM
from criteria import Criteria

def main():
    st.title('FCM Algorithm Demo')

    Y = np.zeros((100, 2))
    
    st.sidebar.header('Input Data and Parameters')

    uploaded_file = st.sidebar.file_uploader("Upload Data File", type=['csv', 'txt'])
    if uploaded_file is not None:
        # Read the uploaded file
        Y = np.loadtxt(uploaded_file, delimiter=',')
        st.sidebar.write('Uploaded Data:')
        st.sidebar.write(Y.shape[0], ' items.')
        print('Uploading data')
        print(Y.shape[0])
        st.sidebar.write(Y.T)
        print(Y)
    else:
        st.sidebar.write('Please upload your data.')

    clusters = st.sidebar.number_input('Number of Clusters', min_value=2, max_value=100, value=2)
    m = st.sidebar.number_input('Fuzziness Coefficient (m)', min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    eps = st.sidebar.number_input('Epsilon', min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    lmax = st.sidebar.number_input('Maximum number of iterations', min_value=10, max_value=100, value=50)

    st.header('Output')

    o = FCM(data=Y, clusters=clusters, m=m, eps=eps, lmax=lmax)

    if st.button('Run Algorithm'):
        print('Running Algorithm')
        updated_u = o.loop()
        o.update_cluster_members()
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
        
        # Clusters members
        for i in range(clusters):
            st.write(f'Cluster {i+1} members:')
            for j in range(o.Y.shape[0]):
                if o.members[j] == i:
                    formatted_data = ", ".join([f"{val:.2f}" for val in o.Y[j]])
                    st.write(f'Point {j+1}: {formatted_data}')
                


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