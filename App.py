import shap
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

@st.cache_resource
def load_model():
    return joblib.load('Models/catboost_modelV2(89.84).pkl')




model = load_model()




edu_map  = {'High School': 0, 'Some College': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
size_map = {'1-10': 0, '11-50': 1, '51-200': 2, '201-1000': 3, '1001-5000': 4, '5000+': 5}
count_map = {'Germany' : 0,'UK' : 1, 'USA' : 2, 'Canada' : 3, 'India' : 4, 'France' : 5, 'Japan' : 6, 'Australia' : 7, 'Brazil' : 8, 'Singapore' : 9 }

language_order  = ['C#', 'C++', 'Go', 'Java', 'JavaScript', 'PHP', 'Python', 'Ruby', 'Rust', 'Swift']
framework_order = ['ASP.NET', 'Angular', 'Django', 'Express', 'Flask', 'Laravel', 'React', 'Ruby on Rails', 'Spring', 'Vue']
full_order = ['experience', 'country', 'education', 'company_size'] + language_order + framework_order
skill_cols = set(language_order + framework_order)

@st.cache_data
def load_background(n=30):
    df = pd.read_csv('train.csv').sample(n=n, random_state=42).reset_index(drop=True)
    df['education'] = df['education'].map(edu_map)
    df['company_size'] = df['company_size'].map(size_map)
    df['country'] = df['country'].map(count_map)

    lang_lists = df['languages'].str.split(', ')
    fw_lists = df['frameworks'].str.split(', ')
    for lang in language_order:
        df[lang] = lang_lists.apply(lambda x: int(lang in x))
    for fw in framework_order:
        df[fw] = fw_lists.apply(lambda x: int(fw in x))

    return df[full_order]



_background = load_background()
@st.cache_resource
def get_explainer(_model, _background):
    # Wrap predict() + expm1() together so SHAP explains dollars, not log-salary
    def predict_dollars(X):
        X_df = pd.DataFrame(X, columns=full_order)
        log_preds = _model.predict(X_df)
        return np.expm1(log_preds)

    return shap.Explainer(predict_dollars, _background, feature_names=full_order)


explainer = get_explainer(model,_background)

salary_predicted = 0

st.title('Yearly Salary Prediction for IT',text_alignment='center')

number = st.slider("Years of experience:", 1, 100, 50)
country = st.selectbox("Country:", options=count_map,key=2)
education = st.selectbox("Education:", options=['High School', 'Some College', 'Bachelors', 'Masters', 'PhD'],key=3)
company_size = st.selectbox("Company Size:", options=size_map,key=4)
language = st.multiselect("Language:", options=language_order,key=5)
framework = st.multiselect("Framework:", options=framework_order,key=6)


def build_display_items(sv, number, country, education, company_size):

    label_map = {
        'experience': f'Experience: {number} yrs',
        'country': f'Country: {country}',
        'education': f'Education: {education}',
        'company_size': f'Company size: {company_size}',
    }

    items = []
    other_sum = 0.0

    for name, val, dval in zip(sv.feature_names, sv.values, sv.data):
        if name in skill_cols:
            if dval == 1:
                items.append((name, float(val)))
            else:
                other_sum += float(val)
        else:
            items.append((label_map.get(name, name), float(val)))

    if abs(other_sum) > 1e-6:
        items.append(("Other skills (not selected)", other_sum))

    items.sort(key=lambda x: abs(x[1]))
    return items


def plot_salary_impact(sv, number, country, education, company_size):
    items = build_display_items(sv, number, country, education, company_size)
    names = [i[0] for i in items]
    values = [i[1] for i in items]

    colors = ['#1e88e5' if v >= 0 else '#ff0d57' for v in values]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 2))
    bars = ax.barh(names, values, color=colors, height=0.6)

    max_abs = max(abs(v) for v in values) if values else 1
    pad = max_abs * 0.03

    for bar, val in zip(bars, values):
        label = f"+${val:,.0f}" if val >= 0 else f"-${abs(val):,.0f}"
        x = bar.get_width()
        if val >= 0:
            ax.text(x + pad, bar.get_y() + bar.get_height() / 2, label,
                    va='center', ha='left', fontsize=9, color='#333333')
        else:
            ax.text(x - pad, bar.get_y() + bar.get_height() / 2, label,
                    va='center', ha='right', fontsize=9, color='#333333')

    ax.set_xlim(min(values + [0]) - max_abs * 0.3, max(values + [0]) + max_abs * 0.3)
    ax.axvline(0, color='#999999', linewidth=0.8)
    ax.set_xlabel("Impact on predicted salary ($)")

    base = float(np.ravel(sv.base_values)[0])
    predicted = base + float(np.sum(sv.values))
    ax.set_title(f"Baseline avg: ${base:,.0f}   →   Predicted: ${predicted:,.0f}", fontsize=11, pad=14)

    ax.legend(
        handles=[mpatches.Patch(color='#ff0d57', label='Increases salary'),
                 mpatches.Patch(color='#1e88e5', label='Decreases salary')],
        loc='lower right', frameon=False, fontsize=8
    )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


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


    st.subheader("Salary Factors")
    st.caption(
        "Each bar shows how many dollars that factor added or subtracted, "
        "moving from the average salary in the data to this prediction."
    )

    with st.spinner("Explaining prediction..."):
        shap_values = explainer(input_df)

    fig = plot_salary_impact(shap_values[0], number, country, education, company_size)
    st.pyplot(fig)
    plt.close(fig)