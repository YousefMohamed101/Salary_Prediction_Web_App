import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Models/catboost_modelV2(89.84).pkl')

st.title('Salary Prediction for IT',text_alignment='center')

number = st.slider("Years of experience:", 1, 100, 50)
country = st.selectbox("Country:", options=['Germany', 'UK', 'USA', 'Canada', 'India', 'France', 'Japan','Australia', 'Brazil', 'Singapore'],key=2)
education = st.selectbox("Education:", options=['High School', 'Some College', 'Bachelors', 'Masters', 'PhD'],key=3)
company_size = st.selectbox("Company Size:", options=['1-10' , '11-50', '51-200', '201-1000' , '1001-5000''5000+'],key=4)
language = st.multiselect("Language:", options=['Python', 'Java', 'Ruby'],key=5)
framework = st.multiselect("Framework:", options=['Python', 'Java', 'Ruby'],key=6)

st.button("predict")
st.write("You selected:", country)