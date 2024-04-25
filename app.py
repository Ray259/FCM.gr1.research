import streamlit as st
from run_with_uploaded_data import main as app1
from run_with_generated_data import main as app2


apps = {
    "Uploaded dataset": app1,
    "Generated dataset (with result plot)": app2
}
st.sidebar.title('Choose dataset')
selection = st.sidebar.radio("Options", list(apps.keys()))

app = apps[selection]
app()
