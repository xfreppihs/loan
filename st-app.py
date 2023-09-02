import streamlit as st
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import imblearn
import shap

st.write("""
# Loan Default Detector

This app predicts credit card applicants at risk of default.

""")

st.sidebar.header('Configuration')

# index starts at 1
applicant_index = st.sidebar.number_input(
    label="Enter the applicant index:",
    min_value=1,
    step=1
)
st.sidebar.write('current applicant index: ', applicant_index)

threshold = st.sidebar.slider(
    label='Choose the threshold:',
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
)
st.sidebar.write("current threshold: ", threshold)

X_test = pd.read_csv("data/processed/X_test.csv")
applicant = X_test.iloc[[applicant_index-1]]

# Reads in saved classification model
load_clf = pickle.load(open('loan_histgb.pkl', 'rb'))

# Apply model to make predictions
prediction_proba = load_clf.predict_proba(applicant)[:, 1][0]
decision = 'approved' if prediction_proba <= threshold else "rejected"

st.subheader('Prediction')
st.write(
    f"The model predicts applicant **{applicant_index}** has a probability of **{round(prediction_proba, 2)}** to default. Based on the chosen threshold **{threshold}**, this application should be **{decision}**.")

# calculate shap values for the applicant

st.subheader('SHAP values for this prediction')
st.write(
    "E[f(X)] is the mean predicted log-odds of all the test applicants. f(X) is the predicted log-odds of applicant ", applicant_index, ". The waterfall plot shows how each feature contributes to the prediction deviating from the mean.")
st.markdown("<span style='color:red'>It may take 5 min to get the SHAP values when runing the first time.</span>",
            unsafe_allow_html=True)


@st.cache_data
def shap_waterfall(_load_clf):
    observations = _load_clf['preprocessor'].transform(X_test)
    explainer = shap.Explainer(
        _load_clf['classifier'], observations, feature_names=X_test.columns)
    return explainer(observations, check_additivity=False)


shap_values = shap_waterfall(load_clf)
st_shap(shap.plots.waterfall(shap_values[applicant_index-1], max_display=10))
