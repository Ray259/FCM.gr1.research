import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from FCM import FCM
from criteria import Criteria

def main():
    st.title('FCM Algorithm Demo')

    if 'generated_data' not in st.session_state:
        st.session_state['generated_data'] = {'Y': np.zeros((100, 2)), 'generated': False}

    st.sidebar.header('Input Data and Parameters')

    generate_data = st.sidebar.button('Generate Data')
    n_samples = st.sidebar.number_input('Number of samples', value=100)
    if generate_data:        
        st.session_state['generated_data']['Y'] = np.random.randint(100, size=(n_samples, 2))
        st.session_state['generated_data']['generated'] = True

    if st.session_state['generated_data']['generated']:
        st.sidebar.write('Generated Data:')
        st.sidebar.write(st.session_state['generated_data']['Y'].T)

        clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=100, value=2)
        m = st.sidebar.slider('Fuzziness Coefficient (m)', min_value=1.1, max_value=5.0, value=2.0)
        eps = st.sidebar.slider('Epsilon', min_value=0.001, max_value=0.1, value=0.01)
        lmax = st.sidebar.slider('Maximum number of iterations', min_value=10, max_value=100, value=50)

        st.header('Output')

        o = FCM(data=st.session_state['generated_data']['Y'], clusters=clusters, m=m, eps=eps, lmax=lmax)
        if st.button('Run Algorithm'):
            updated_u = o.loop()
            o.update_cluster_members()
            st.write('Final Membership:')
            st.write(updated_u)

            # Plot clusters
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state['generated_data']['Y'][:, 0], st.session_state['generated_data']['Y'][:, 1], c=np.argmax(updated_u, axis=0), cmap='viridis')
            ax.scatter(o.centers[:, 0], o.centers[:, 1], marker='X', s=100, c='red')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('FCM Clustering')
            ax.legend(['Data points','Cluster Centers'], loc='upper right')
            st.pyplot(fig)
            
            # Validity            
            cr = Criteria(o.centers, o.cluster_members, o.members, o.Y.shape[0], clusters, o.data_center, o.Y, o.u, o.m)
            crt = cr.validate()
            for c in crt:
                st.write(c[0] + ':', c[1])

if __name__ == '__main__':
    main()

css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 500px;
        max-width: 800px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)