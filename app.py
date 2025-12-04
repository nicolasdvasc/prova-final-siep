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

percent_income = amount / income if income > 0 else 0

# DataFrame Input
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
        input_processed = input_data.copy()
        for col, le in encoders.items():
            if col in input_processed.columns:
                valor_texto = input_processed[col].astype(str)
                input_processed[col] = le.transform(valor_texto)

        # 2. ESCALONAMENTO
        X_final = scaler.transform(input_processed)

        # 3. PREDIÇÃO
        proba = model.predict_proba(X_final)[:, 1][0]
        
        # Resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Resultado da Análise")
            if proba > 0.5:
                st.error("🛑 ALTO RISCO (REPROVAR)")
                st.metric("Score de Risco", f"{proba:.1%}", delta="Risco Elevado", delta_color="inverse")
                st.markdown("**Ação Recomendada:** Negar crédito.")
            else:
                st.success("✅ BAIXO RISCO (APROVAR)")
                st.metric("Score de Risco", f"{proba:.1%}", delta="Aprovado", delta_color="normal")
                st.markdown("**Ação Recomendada:** Conceder crédito.")

        with col2:
            st.subheader("🔍 Por que este resultado?")
            with st.spinner("Gerando explicação (SHAP)..."):
                try:
                    # Tenta criar o explicador
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_final)
                        base_value = explainer.expected_value
                    except:
                        # Fallback para KernelExplainer
                        explainer = shap.KernelExplainer(model.predict_proba, X_final)
                        shap_values = explainer.shap_values(X_final)
                        base_value = explainer.expected_value

                    # --- CORREÇÃO DE DIMENSÃO (O SEU ERRO ESTAVA AQUI) ---
                    # O objetivo é ficar apenas com um array 1D de tamanho 11 (features)
                    
                    vals = shap_values
                    
                    # Caso 1: Se for lista (ex: [matriz_classe0, matriz_classe1]), pega a classe 1
                    if isinstance(vals, list):
                        vals = vals[1]
                        if isinstance(base_value, list):
                            base_value = base_value[1]

                    # Caso 2: Se for 3D (1 amostra, 11 features, 2 classes) -> Pega amostra 0 e classe 1
                    if len(vals.shape) == 3:
                        vals = vals[0, :, 1]
                    
                    # Caso 3: Se for 2D (1 amostra, 11 features) -> Pega amostra 0
                    elif len(vals.shape) == 2 and vals.shape[0] == 1:
                        vals = vals[0]
                        
                    # Caso 4 (O SEU ERRO): Se for 2D (11 features, 2 classes) -> Pega todas features da classe 1
                    elif len(vals.shape) == 2 and vals.shape[1] == 2:
                        vals = vals[:, 1]

                    # Garante que o base_value seja um número único (float)
                    if hasattr(base_value, 'shape') and len(base_value.shape) > 0:
                         base_value = base_value[0] if len(base_value) == 1 else base_value[1]
                    
                    # --- CRIAÇÃO DO GRÁFICO ---
                    explanation = shap.Explanation(
                        values=vals,
                        base_values=base_value,
                        data=input_data.iloc[0].values,
                        feature_names=input_data.columns
                    )

                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(explanation, show=False)
                    st.pyplot(fig, bbox_inches='tight')

                except Exception as e:
                    st.warning(f"Não foi possível gerar o gráfico SHAP. Erro: {e}")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
