
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CrediFast Risk", layout="wide")
st.title("CrediFast - Análise de Risco de Crédito")
st.markdown("**Aluno:** Nícolas Duarte Vasconcellos (ID: 200042343)")

@st.cache_resource
def load_data():
    try:
        model = joblib.load('modelo_credifast.pkl')
        prep = joblib.load('preprocessor.pkl')
        return model, prep
    except:
        return None, None

model, preprocessor = load_data()

if model is None:
    st.error("Modelos não encontrados. Faça upload dos arquivos .pkl gerados no Colab.")
    st.stop()

st.sidebar.header("Dados do Cliente")
age = st.sidebar.number_input("Idade", 18, 100, 30)
income = st.sidebar.number_input("Renda Anual", 1000, 1000000, 50000)
home = st.sidebar.selectbox("Moradia", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
emp_len = st.sidebar.number_input("Anos de Emprego", 0, 50, 5)
intent = st.sidebar.selectbox("Motivo", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
grade = st.sidebar.selectbox("Grau (Grade)", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
amount = st.sidebar.number_input("Valor Empréstimo", 100, 50000, 10000)
rate = st.sidebar.number_input("Taxa Juros (%)", 5.0, 30.0, 10.0)
default_hist = st.sidebar.selectbox("Inadimplência Prévia?", ['N', 'Y'])
cred_hist = st.sidebar.number_input("Anos de Histórico", 2, 30, 4)

input_data = pd.DataFrame([{
    'person_age': age, 'person_income': income, 'person_home_ownership': home,
    'person_emp_length': emp_len, 'loan_intent': intent, 'loan_grade': grade,
    'loan_amnt': amount, 'loan_int_rate': rate,
    'loan_percent_income': amount/income if income else 0,
    'cb_person_default_on_file': default_hist, 'cb_person_cred_hist_length': cred_hist
}])

if st.button("Avaliar Risco"):
    X_processed = preprocessor.transform(input_data)
    proba = model.predict_proba(X_processed)[:, 1][0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidade de Default", f"{proba:.1%}")
        if proba > 0.5:
            st.error("ALTO RISCO - REPROVAR")
        else:
            st.success("BAIXO RISCO - APROVAR")
    with col2:
        st.write("Explicabilidade:")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar", feature_names=preprocessor.get_feature_names_out(), show=False)
        st.pyplot(fig)
