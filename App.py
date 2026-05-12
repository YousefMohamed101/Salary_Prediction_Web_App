import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fontTools.misc.cython import returns

model = joblib.load('Models/catboost_modelV2(89.84).pkl')

edu_map  = {'High School': 0, 'Some College': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
size_map = {'1-10': 0, '11-50': 1, '51-200': 2, '201-1000': 3, '1001-5000': 4, '5000+': 5}
count_map = {'Germany' : 0,'UK' : 1, 'USA' : 2, 'Canada' : 3, 'India' : 4, 'France' : 5, 'Japan' : 6, 'Australia' : 7, 'Brazil' : 8, 'Singapore' : 9 }

language_order  = ['C#', 'C++', 'Go', 'Java', 'JavaScript', 'PHP', 'Python', 'Ruby', 'Rust', 'Swift']
framework_order = ['ASP.NET', 'Angular', 'Django', 'Express', 'Flask', 'Laravel', 'React', 'Ruby on Rails', 'Spring', 'Vue']
full_order = ['experience', 'country', 'education', 'company_size'] + language_order + framework_order

salary_predicted = 0

st.title('Yearly Salary Prediction for IT',text_alignment='center')

number = st.slider("Years of experience:", 1, 100, 50)
country = st.selectbox("Country:", options=count_map,key=2)
education = st.selectbox("Education:", options=['High School', 'Some College', 'Bachelors', 'Masters', 'PhD'],key=3)
company_size = st.selectbox("Company Size:", options=size_map,key=4)
language = st.multiselect("Language:", options=language_order,key=5)
framework = st.multiselect("Framework:", options=framework_order,key=6)




if st.button("predict"):
    data = {col: 0 for col in full_order}

    if language == [] or framework == []:
        st.warning("Please select at least one language and one framework.")
        st.stop()

    data['experience'] = number
    data['country'] = count_map[country]
    data['education'] = edu_map[education]
    data['company_size'] = size_map[company_size]
    for lang in language:
        data[lang] = 1
    for frame in framework:
        data[frame] = 1


    input_df = pd.DataFrame([data])[full_order]
    prediction = model.predict(input_df)[0]
    salary_predicted = np.expm1(prediction)


st.write(f"Predicted yearly salary: {salary_predicted:.2f}$")