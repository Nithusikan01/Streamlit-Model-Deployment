import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# App config
st.set_page_config(page_title="Iris Flower Classification", layout="wide")
st.title("🌸 Iris Flower Classification App")

# Sidebar
with st.sidebar:
    st.header("📄 Data Requirements")
    st.caption('Upload a CSV with **4 numeric columns**. Column names don’t matter.')
    with st.expander("🔍 Expected Format"):
        st.markdown("""
        - UTF-8 encoding  
        - Comma-separated  
        - Dot (.) as decimal  
        - First row as header
        """)
    st.divider()
    st.caption("<p style='text-align:center'>Developed by Ntx</p>", unsafe_allow_html=True)

# Button logic
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started!", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] != 4:
                st.error("❌ Please upload a CSV with exactly 4 columns.")
            else:
                st.header("📊 Uploaded Data Preview")
                st.write(df.head())

                # Load model
                try:
                    model = joblib.load("models/random_forest_classifier.joblib")
                except FileNotFoundError:
                    st.error("🚫 Model file not found! Make sure 'models/random_forest_classifier.joblib' exists in your repo.")
                else:
                    pred = model.predict_proba(df)
                    pred_df = pd.DataFrame(pred, columns=['setosa_probability', 'versicolor_probability', 'virginica_probability'])

                    st.header("🎯 Prediction Results")
                    st.write(pred_df.head())

                    csv_data = pred_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions", data=csv_data, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
