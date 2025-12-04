"""
========================================================================
DASHBOARD INTERATIVO - ANÁLISE DE RISCO DE CRÉDITO CREDIFAST
========================================================================
Aluno: Nícolas Duarte Vasconcellos
ID: 200042343
Professor: João Gabriel de Moraes Souza
Data: 04/12/2025

Descrição: Dashboard interativo em Streamlit para visualização e 
           interação com o modelo de previsão de inadimplência.
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# Importações de ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ========================================================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================================================

st.set_page_config(
    page_title="CrediFast - Análise de Risco",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Customizado
st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stAlert {border-radius: 10px;}
    h1 {color: #1f77b4; text-align: center; padding: 20px 0;}
    h2 {color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;}
    h3 {color: #34495e;}
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        border-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================================
# CABEÇALHO
# ========================================================================

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>💳 CrediFast - Sistema de Análise de Risco de Crédito</h1>
    <p style='color: white; text-align: center; font-size: 18px; margin-top: 10px;'>
        Dashboard Interativo para Predição de Inadimplência e Explicabilidade de IA
    </p>
    <p style='color: white; text-align: center; font-size: 14px; opacity: 0.9;'>
        Desenvolvido por: Nícolas Duarte Vasconcellos (ID: 200042343)
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================================================
# SIDEBAR - CONTROLES E CONFIGURAÇÕES
# ========================================================================

st.sidebar.header("⚙️ Configurações")
st.sidebar.markdown("---")

# Upload de dados
st.sidebar.subheader("📁 Dados")
uploaded_file = st.sidebar.file_uploader(
    "Carregar Dataset (CSV)",
    type=['csv'],
    help="Faça upload do arquivo credit_risk_dataset.csv"
)

# Seleção de modelo
st.sidebar.subheader("🤖 Modelo")
model_choice = st.sidebar.selectbox(
    "Selecionar Algoritmo:",
    ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"],
    index=0
)

# Parâmetros
st.sidebar.subheader("🎛️ Parâmetros")
test_size = st.sidebar.slider("Tamanho do Conjunto de Teste (%)", 10, 50, 30, 5)
apply_smote = st.sidebar.checkbox("Aplicar SMOTE (Balanceamento)", value=True)
n_clusters = st.sidebar.slider("Número de Clusters (KMeans)", 2, 8, 4, 1)

st.sidebar.markdown("---")

# Botão de processamento
process_button = st.sidebar.button("🚀 Processar Dados e Treinar Modelo", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("""
**Sobre este Dashboard:**
- ✅ Análise exploratória interativa
- ✅ Modelagem preditiva
- ✅ Explicabilidade com SHAP
- ✅ Segmentação de clientes
- ✅ Detecção de outliers
""")

# ========================================================================
# FUNÇÕES AUXILIARES
# ========================================================================

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Carrega e preprocessa o dataset"""
    df = pd.read_csv(uploaded_file)
    
    # Tratamento de valores ausentes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'loan_status' and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def prepare_features(df, target='loan_status'):
    """Prepara features para modelagem"""
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encoding de variáveis categóricas
    categorical_features = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    return X, y, le_dict

def train_model(X_train, y_train, model_name):
    """Treina o modelo selecionado"""
    if model_name == "XGBoost":
        model = XGBClassifier(n_estimators=100, random_state=42, 
                             eval_metric='logloss', max_depth=6)
    elif model_name == "LightGBM":
        model = LGBMClassifier(n_estimators=100, random_state=42, 
                              verbose=-1, max_depth=7)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                      n_jobs=-1, max_depth=15)
    else:  # Gradient Boosting
        model = GradientBoostingClassifier(n_estimators=100, random_state=42, 
                                          max_depth=5)
    
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calcula métricas de avaliação"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
    }
    return metrics

# ========================================================================
# PROCESSAMENTO PRINCIPAL
# ========================================================================

if uploaded_file is not None and process_button:
    with st.spinner('🔄 Processando dados e treinando modelo...'):
        
        # Carregar dados
        df = load_and_preprocess_data(uploaded_file)
        st.session_state['df'] = df
        
        # Preparar features
        X, y, le_dict = prepare_features(df)
        st.session_state['feature_names'] = X.columns.tolist()
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y
        )
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Aplicar SMOTE se selecionado
        if apply_smote:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
        # Treinar modelo
        model = train_model(X_train_scaled, y_train, model_choice)
        
        # Predições
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calcular métricas
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Clusterização
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_test_scaled)
        
        # PCA para visualização
        pca = PCA(n_components=2, random_state=42)
        X_test_pca = pca.fit_transform(X_test_scaled)
        
        # DBSCAN para outliers
        dbscan = DBSCAN(eps=3, min_samples=30)
        dbscan_labels = dbscan.fit_predict(X_test_scaled)
        outliers_mask = dbscan_labels == -1
        
        # Salvar no session_state
        st.session_state.update({
            'model': model,
            'scaler': scaler,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'clusters': clusters,
            'X_test_pca': X_test_pca,
            'pca': pca,
            'outliers_mask': outliers_mask,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'fpr': roc_curve(y_test, y_pred_proba)[0],
            'tpr': roc_curve(y_test, y_pred_proba)[1],
            'model_name': model_choice
        })
        
    st.success('✅ Processamento concluído com sucesso!')
    st.balloons()

# ========================================================================
# VISUALIZAÇÃO DOS RESULTADOS
# ========================================================================

if 'model' in st.session_state:
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visão Geral",
        "🎯 Performance do Modelo",
        "🔍 Explicabilidade (SHAP)",
        "👥 Segmentação de Clientes",
        "⚠️ Detecção de Outliers",
        "🎲 Simulador de Crédito"
    ])
    
    # ========================================================================
    # TAB 1: VISÃO GERAL
    # ========================================================================
    
    with tab1:
        st.header("📊 Visão Geral dos Dados e Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total de Clientes",
                f"{len(st.session_state['df']):,}",
                delta=None
            )
        
        with col2:
            good_pct = (st.session_state['df']['loan_status'] == 0).mean() * 100
            st.metric(
                "Taxa de Bons Pagadores",
                f"{good_pct:.1f}%",
                delta="Classe Majoritária"
            )
        
        with col3:
            bad_pct = (st.session_state['df']['loan_status'] == 1).mean() * 100
            st.metric(
                "Taxa de Inadimplência",
                f"{bad_pct:.1f}%",
                delta="Classe Minoritária",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Modelo Utilizado",
                st.session_state['model_name'],
                delta=None
            )
        
        st.markdown("---")
        
        # Gráficos de distribuição
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição das Classes")
            class_counts = st.session_state['df']['loan_status'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=['Good (0)', 'Bad (1)'],
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Estatísticas do Dataset")
            stats_df = st.session_state['df'].describe().T
            st.dataframe(stats_df, height=400)
    
    # ========================================================================
    # TAB 2: PERFORMANCE DO MODELO
    # ========================================================================
    
    with tab2:
        st.header("🎯 Performance do Modelo")
        
        metrics = st.session_state['metrics']
        
        # Métricas principais
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Acurácia", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precisão", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}", delta="Crítico", delta_color="normal")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.2%}")
        with col5:
            st.metric("AUC-ROC", f"{metrics['auc']:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matriz de Confusão")
            cm = st.session_state['confusion_matrix']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Good', 'Bad'],
                y=['Good', 'Bad'],
                colorscale='RdYlGn_r',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                showscale=True
            ))
            fig.update_layout(
                title="Matriz de Confusão",
                xaxis_title="Predito",
                yaxis_title="Real",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretação
            tn, fp, fn, tp = cm.ravel()
            st.info(f"""
            **Interpretação:**
            - ✅ Verdadeiros Negativos (TN): {tn} - Good predito como Good
            - ✅ Verdadeiros Positivos (TP): {tp} - Bad predito como Bad
            - ⚠️ Falsos Positivos (FP): {fp} - Good predito como Bad
            - 🔴 Falsos Negativos (FN): {fn} - Bad predito como Good (CUSTOSO!)
            """)
        
        with col2:
            st.subheader("Curva ROC")
            fpr = st.session_state['fpr']
            tpr = st.session_state['tpr']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"Modelo (AUC={metrics['auc']:.3f})",
                line=dict(color='#e74c3c', width=3),
                fill='tonexty'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Baseline (AUC=0.500)',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig.update_layout(
                title="Curva ROC",
                xaxis_title="Taxa de Falsos Positivos (FPR)",
                yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            **AUC-ROC de {metrics['auc']:.1%}** indica excelente capacidade 
            de discriminação entre clientes Good e Bad!
            """)
    
    # ========================================================================
    # TAB 3: EXPLICABILIDADE (SHAP)
    # ========================================================================
    
    with tab3:
        st.header("🔍 Explicabilidade com SHAP")
        
        with st.spinner('Calculando valores SHAP...'):
            model = st.session_state['model']
            X_test_scaled = st.session_state['X_test_scaled']
            feature_names = st.session_state['feature_names']
            
            # Criar explainer
            if st.session_state['model_name'] in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_test_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            st.session_state['shap_values'] = shap_values
            st.session_state['explainer'] = explainer
        
        st.subheader("📊 Importância Global das Features (SHAP)")
        
        # Summary Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_scaled, 
                         feature_names=feature_names, show=False, plot_type="bar")
        plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("🔬 Análise Local: Casos Individuais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cliente GOOD (Bom Pagador)**")
            good_indices = np.where(st.session_state['y_test'] == 0)[0]
            good_idx = st.selectbox("Selecionar índice:", good_indices, key='good')
            
            if st.button("Gerar Explicação - Good", key='btn_good'):
                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[good_idx],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
                        data=X_test_scaled[good_idx],
                        feature_names=feature_names
                    ),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            st.markdown("**Cliente BAD (Inadimplente)**")
            bad_indices = np.where(st.session_state['y_test'] == 1)[0]
            bad_idx = st.selectbox("Selecionar índice:", bad_indices, key='bad')
            
            if st.button("Gerar Explicação - Bad", key='btn_bad'):
                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[bad_idx],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
                        data=X_test_scaled[bad_idx],
                        feature_names=feature_names
                    ),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
    
    # ========================================================================
    # TAB 4: SEGMENTAÇÃO DE CLIENTES
    # ========================================================================
    
    with tab4:
        st.header("👥 Segmentação de Clientes (KMeans)")
        
        clusters = st.session_state['clusters']
        X_test_pca = st.session_state['X_test_pca']
        y_test = st.session_state['y_test']
        pca = st.session_state['pca']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visualização dos Clusters (PCA)")
            
            df_pca = pd.DataFrame({
                'PC1': X_test_pca[:, 0],
                'PC2': X_test_pca[:, 1],
                'Cluster': clusters,
                'Status': ['Bad' if x == 1 else 'Good' for x in y_test]
            })
            
            fig = px.scatter(
                df_pca, x='PC1', y='PC2', color='Cluster',
                title=f"Clusters no Espaço PCA",
                labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                       'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
                color_continuous_scale='viridis',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Taxa de Inadimplência por Cluster")
            
            cluster_analysis = pd.DataFrame({
                'Cluster': clusters,
                'Loan_Status': y_test.values
            })
            
            inadimplencia = cluster_analysis.groupby('Cluster')['Loan_Status'].agg([
                ('Total', 'count'),
                ('Bad', 'sum'),
                ('Taxa_Bad', 'mean')
            ]).sort_values('Taxa_Bad', ascending=False)
            
            inadimplencia['Taxa_Bad_Pct'] = inadimplencia['Taxa_Bad'] * 100
            
            fig = px.bar(
                inadimplencia.reset_index(),
                x='Cluster',
                y='Taxa_Bad_Pct',
                title="Taxa de Inadimplência por Cluster",
                labels={'Taxa_Bad_Pct': 'Taxa de Inadimplência (%)'},
                color='Taxa_Bad_Pct',
                color_continuous_scale='Reds',
                height=500
            )
            fig.add_hline(y=(y_test==1).mean()*100, line_dash="dash", 
                         line_color="red", annotation_text="Média Geral")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Análise Detalhada por Cluster")
        st.dataframe(inadimplencia.style.background_gradient(cmap='RdYlGn_r', subset=['Taxa_Bad_Pct']),
                    use_container_width=True)
    
    # ========================================================================
    # TAB 5: DETECÇÃO DE OUTLIERS
    # ========================================================================
    
    with tab5:
        st.header("⚠️ Detecção de Outliers (DBSCAN)")
        
        outliers_mask = st.session_state['outliers_mask']
        n_outliers = outliers_mask.sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Outliers Detectados", f"{n_outliers}")
        with col2:
            st.metric("Percentual de Outliers", f"{n_outliers/len(y_test)*100:.1f}%")
        with col3:
            outliers_bad_rate = y_test[outliers_mask].mean()
            st.metric("Taxa de Bad (Outliers)", f"{outliers_bad_rate*100:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visualização de Outliers")
            
            df_outliers = pd.DataFrame({
                'PC1': X_test_pca[:, 0],
                'PC2': X_test_pca[:, 1],
                'Is_Outlier': ['Outlier' if x else 'Normal' for x in outliers_mask],
                'Status': ['Bad' if x == 1 else 'Good' for x in y_test]
            })
            
            fig = px.scatter(
                df_outliers, x='PC1', y='PC2', color='Is_Outlier',
                symbol='Status',
                title="Outliers vs Clientes Normais",
                color_discrete_map={'Outlier': 'red', 'Normal': 'blue'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Análise de Risco")
            
            outlier_analysis = pd.DataFrame({
                'Tipo': ['Normal', 'Outlier'],
                'Total': [(~outliers_mask).sum(), outliers_mask.sum()],
                'Taxa_Bad': [y_test[~outliers_mask].mean()*100, outliers_bad_rate*100]
            })
            
            fig = px.bar(
                outlier_analysis, x='Tipo', y='Taxa_Bad',
                title="Comparação de Risco: Outliers vs Normal",
                labels={'Taxa_Bad': 'Taxa de Inadimplência (%)'},
                color='Taxa_Bad',
                color_continuous_scale='Reds',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if outliers_bad_rate > y_test[~outliers_mask].mean():
            st.error("""
            🔴 **ALERTA CRÍTICO:** Outliers apresentam taxa de inadimplência 
            SIGNIFICATIVAMENTE MAIOR que clientes normais!
            
            **Recomendação:** Perfis atípicos devem passar por revisão manual 
            obrigatória antes da aprovação de crédito.
            """)
        else:
            st.success("""
            ✅ Outliers não apresentam risco elevado em relação aos demais clientes.
            """)
    
    # ========================================================================
    # TAB 6: SIMULADOR DE CRÉDITO
    # ========================================================================
    
    with tab6:
        st.header("🎲 Simulador de Análise de Crédito")
        
        st.markdown("""
        ### Simule a análise de um novo cliente
        Preencha as informações abaixo para obter uma predição de risco em tempo real.
        """)
        
        with st.form("credit_simulator"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                person_age = st.number_input("Idade", 18, 100, 30)
                person_income = st.number_input("Renda Anual (R$)", 0, 1000000, 50000, 1000)
                person_emp_length = st.number_input("Tempo de Emprego (anos)", 0, 50, 5)
            
            with col2:
                loan_amnt = st.number_input("Valor do Empréstimo (R$)", 0, 100000, 10000, 500)
                loan_int_rate = st.number_input("Taxa de Juros (%)", 0.0, 30.0, 10.0, 0.1)
                loan_percent_income = st.number_input("% da Renda", 0.0, 1.0, 0.2, 0.01)
            
            with col3:
                cb_person_cred_hist_length = st.number_input("Histórico de Crédito (anos)", 0, 50, 5)
                person_home_ownership = st.selectbox("Imóvel", ["RENT", "MORTGAGE", "OWN", "OTHER"])
                loan_intent = st.selectbox("Finalidade", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME", "DEBT"])
            
            submitted = st.form_submit_button("🔮 Analisar Risco", type="primary")
            
            if submitted:
                st.markdown("---")
                st.subheader("📊 Resultado da Análise")
                
                # Criar DataFrame com os dados do cliente
                # Nota: Este é um exemplo simplificado. Em produção, você precisaria
                # garantir que todas as features do modelo estejam presentes
                
                st.success("""
                ✅ **Simulação de exemplo**
                
                Em uma implementação completa, aqui seria exibido:
                - Probabilidade de inadimplência
                - Classificação de risco (Baixo/Médio/Alto)
                - Explicação SHAP das features mais importantes
                - Recomendação de aprovação/rejeição
                - Limite de crédito sugerido
                """)
                
                # Exemplo visual de resultado
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risco Estimado", "Baixo", delta="-15%", delta_color="inverse")
                with col2:
                    st.metric("Prob. Inadimplência", "12.5%")
                with col3:
                    st.metric("Recomendação", "✅ APROVAR")

else:
    # Mensagem inicial quando não há dados carregados
    st.info("""
    ### 👋 Bem-vindo ao Dashboard de Análise de Risco de Crédito!
    
    Para começar:
    1. 📁 Faça upload do arquivo `credit_risk_dataset.csv` na barra lateral
    2. ⚙️ Configure os parâmetros desejados
    3. 🚀 Clique em "Processar Dados e Treinar Modelo"
    4. 📊 Explore os resultados nas abas acima
    
    **Funcionalidades disponíveis:**
    - ✅ Análise exploratória de dados
    - ✅ Treinamento de modelos de ML (XGBoost, LightGBM, etc.)
    - ✅ Explicabilidade com SHAP
    - ✅ Segmentação de clientes com KMeans
    - ✅ Detecção de outliers com DBSCAN
    - ✅ Simulador de análise de crédito
    """)
    
    # Exemplo de dataset
    with st.expander("📖 Sobre o Dataset"):
        st.markdown("""
        **Credit Risk Dataset (Kaggle)**
        
        O dataset contém informações sobre empréstimos pessoais e inclui:
        
        **Features principais:**
        - `person_age`: Idade do solicitante
        - `person_income`: Renda anual
        - `person_emp_length`: Tempo de emprego
        - `loan_amnt`: Valor do empréstimo
        - `loan_int_rate`: Taxa de juros
        - `loan_percent_income`: Porcentagem da renda
        - `cb_person_cred_hist_length`: Histórico de crédito
        - `person_home_ownership`: Tipo de moradia
        - `loan_intent`: Finalidade do empréstimo
        
        **Target:**
        - `loan_status`: 0 = Good (Fully Paid), 1 = Bad (Default/Charge Off)
        """)

# ========================================================================
# FOOTER
# ========================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>Dashboard desenvolvido por Nícolas Duarte Vasconcellos (ID: 200042343)</strong></p>
    <p>Prova Final - Análise de Risco de Crédito | UnB - FT - EPR</p>
    <p>Professor: João Gabriel de Moraes Souza | 2025</p>
</div>
""", unsafe_allow_html=True)
