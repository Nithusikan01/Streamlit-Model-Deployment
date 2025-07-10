import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Iris Flower Classification", layout='wide')
st.title("Iris Flower Classification App")

with st.sidebar:
    st.header("Data Requirements")
    st.caption('To inference the model you need to upload a dataframe in csv format with four columns/features (columns names are not important)')
    with st.expander("Data format"):
        st.markdown(' - utf-8')
        st.markdown(' - separated by commas')
        st.markdown(' - delimitered by "."')
        st.markdown(' - first row as header')
    st.divider()
    st.caption("<p style = 'text-align:center'>Developed by Ntx.</p>", unsafe_allow_html=True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started!", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
       df = pd.read_csv(uploaded_file, low_memory=True)
       st.header("Uploaded Data Preview")
       st.write(df.head())
       model = joblib.load('random_forest_classifier.joblib')
       pred = model.predict_proba(df)
       pred = pd.DataFrame(pred, columns = ['setosa_probability', 'versicolor_probability', 'virginica_probability'])
       st.header("Prediction Results")
       st.write(pred.head())
       
       pred = pred.to_csv(index=False).encode('utf-8')
       st.download_button(
           label="Download Predictions",
           data=pred,
           file_name='predictions.csv',
           mime='text/csv',
           key='download-csv'
       )
