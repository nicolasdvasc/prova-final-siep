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
        prediction = model.predict(X_final)[0]
        
        # Resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Resultado da Análise")
            if proba > 0.5:
                st.error("🛑 ALTO RISCO (REPROVAR)")
                st.metric("Score de Risco", f"{proba:.1%}", delta="Risco Elevado", delta_color="inverse")
                st.markdown("**Ação Recomendada:** Negar crédito ou exigir garantias.")
            else:
                st.success("✅ BAIXO RISCO (APROVAR)")
                st.metric("Score de Risco", f"{proba:.1%}", delta="Aprovado", delta_color="normal")
                st.markdown("**Ação Recomendada:** Conceder crédito.")

        with col2:
            st.subheader("🔍 Por que este resultado?")
            with st.spinner("Gerando gráfico detalhado (Waterfall)..."):
                try:
                    # --- CÁLCULO DO SHAP ---
                    # Tenta TreeExplainer (Rápido), se falhar vai de KernelExplainer (Lento/Genérico)
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_final)
                        expected_value = explainer.expected_value
                    except Exception:
                        explainer = shap.KernelExplainer(model.predict_proba, X_final)
                        shap_values = explainer.shap_values(X_final)
                        expected_value = explainer.expected_value

                    # --- TRATAMENTO DOS DADOS PARA O GRÁFICO ---
                    # O SHAP retorna listas ou arrays dependendo do modelo. 
                    # Precisamos garantir que estamos pegando os números certos.
                    
                    # Se for lista (comum em classificação binária), pega a classe 1 (Risco/Bad)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                        expected_value = expected_value[1]
                    
                    # Se tiver dimensão extra (ex: [[...]]), remove para ficar 1D ([...])
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    # Cria um objeto de Explicação robusto
                    # Isso "cola" os valores matemáticos com os nomes das colunas e os dados originais
                    explanation = shap.Explanation(
                        values=shap_values,
                        base_values=expected_value,
                        data=input_data.iloc[0].values, # Mostra os dados originais no gráfico (mais bonito)
                        feature_names=input_data.columns
                    )

                    # --- PLOTAGEM ---
                    # Cria a figura explicitamente
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Desenha o Waterfall
                    shap.plots.waterfall(explanation, show=False)
                    
                    # Exibe no Streamlit
                    st.pyplot(fig, bbox_inches='tight')
                    
                    st.caption("Gráfico Waterfall: Mostra como cada característica empurrou a nota do cliente para cima (risco) ou para baixo (segurança) a partir da média.")

                except Exception as e:
                    st.warning(f"Não foi possível gerar o gráfico SHAP. Detalhe do erro: {e}")

    except Exception as e:
        st.error(f"Erro crítico no processamento: {e}")
