import streamlit as st
import pandas as pd
import joblib

model = joblib.load('catboost_modelV2(89.84).pkl')

st.title('Salary Prediction for IT')


exp = st.multiselect("Years of experience:", options=['Python', 'Java', 'Ruby'],max_selections=1,key=1)
country = st.multiselect("Country:", options=['Python', 'Java', 'Ruby'],key=2)
education = st.multiselect("Education:", options=['Python', 'Java', 'Ruby'],key=3)
company_size = st.multiselect("Education:", options=['Python', 'Java', 'Ruby'],key=4)

st.write("You selected:", exp)