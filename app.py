import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Configuração da Página ---
st.set_page_config(page_title="CrediFast Risk", layout="wide")

st.title("🏦 CrediFast: Sistema de Análise de Risco")
st.markdown("""
**Aluno:** Nícolas Duarte Vasconcellos | **ID:** 200042343
**Disciplina:** Análise de Dados (SIEP)
---
""")

# --- Carregamento dos Arquivos ---
@st.cache_resource
def load_data():
    try:
        # Carregamos Modelo, Scaler (Preprocessador) e Encoders (Tradutores)
        model = joblib.load('modelo_credifast.pkl')
        scaler = joblib.load('preprocessor.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, scaler, encoders
    except FileNotFoundError:
        return None, None, None

model, scaler, encoders = load_data()

if model is None:
    st.error("❌ Erro de Arquivos: Faltam arquivos .pkl no repositório.")
    st.info("Certifique-se de que subiu: modelo_credifast.pkl, preprocessor.pkl e encoders.pkl")
    st.stop()

# --- Sidebar: Dados do Cliente ---
st.sidebar.header("📝 Dados do Solicitante")

age = st.sidebar.number_input("Idade", 18, 100, 25)
income = st.sidebar.number_input("Renda Anual ($)", 1000, 1000000, 50000)
home = st.sidebar.selectbox("Moradia", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
emp_len = st.sidebar.number_input("Anos de Emprego", 0, 50, 5)
intent = st.sidebar.selectbox("Finalidade", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
grade = st.sidebar.selectbox("Grau de Risco", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
amount = st.sidebar.number_input("Valor Empréstimo ($)", 100, 50000, 10000)
rate = st.sidebar.number_input("Taxa de Juros (%)", 4.0, 25.0, 10.0)
default = st.sidebar.selectbox("Já teve Inadimplência?", ['N', 'Y'])
cred_hist = st.sidebar.number_input("Histórico de Crédito (anos)", 2, 30, 4)

# Lógica igual ao notebook
percent_income = amount / income if income > 0 else 0

# Criar DataFrame com as colunas na ORDEM EXATA do treinamento
input_data = pd.DataFrame([{
    'person_age': age,
    'person_income': income,
    'person_home_ownership': home,
    'person_emp_length': emp_len,
    'loan_intent': intent,
    'loan_grade': grade,
    'loan_amnt': amount,
    'loan_int_rate': rate,
    'loan_percent_income': percent_income,
    'cb_person_default_on_file': default,
    'cb_person_cred_hist_length': cred_hist
}])

# --- Botão de Cálculo ---
if st.button("🚀 Calcular Risco"):
    try:
        # 1. TRADUÇÃO (Label Encoding)
        # Convertemos texto para números usando os encoders salvos
        input_processed = input_data.copy()
        
        for col, le in encoders.items():
            if col in input_processed.columns:
                # O encoder espera um array, transformamos o valor único
                valor_texto = input_processed[col].astype(str)
                input_processed[col] = le.transform(valor_texto)

        # 2. ESCALONAMENTO (StandardScaler)
        # Agora que tudo é número, podemos aplicar a matemática
        X_final = scaler.transform(input_processed)

        # 3. PREDIÇÃO
        proba = model.predict_proba(X_final)[:, 1][0]
        
        # Resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score de Risco")
            if proba > 0.5:
                st.error("🛑 ALTO RISCO (REPROVAR)")
                st.metric("Probabilidade de Calote", f"{proba:.1%}", delta="-Alto Risco")
            else:
                st.success("✅ BAIXO RISCO (APROVAR)")
                st.metric("Probabilidade de Calote", f"{proba:.1%}", delta="Baixo Risco")

        with col2:
            st.subheader("Explicabilidade (SHAP)")
            with st.spinner("Analisando motivos..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_final)
                
                # Ajuste para formatos diferentes de retorno do SHAP
                if isinstance(shap_values, list):
                    vals = shap_values[1]
                else:
                    vals = shap_values
                
                # Gráfico
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.summary_plot(vals, input_processed, plot_type="bar", 
                                feature_names=input_data.columns, show=False)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
