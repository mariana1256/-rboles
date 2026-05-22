import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(page_title="Iris ID3 Classifier", layout="wide")

# Función para cargar datos
@st.cache_data
def load_data():
    url = "https://www.openml.org/data/get_csv/61/dataset_61_iris.arff"
    df = pd.read_csv(url)
    return df

df = load_data()

# Título y navegación
st.title("🌿 Clasificador de Flores Iris - Algoritmo ID3")
menu = st.sidebar.selectbox("Navegación", ["Presentación", "Exploración de Datos", "Modelado y Evaluación", "Predicción Interactiva"])

if menu == "Presentación":
    st.header("Sobre el Proyecto")
    st.write("""
    Esta aplicación presenta un modelo de clasificación basado en el algoritmo **ID3 (Iterative Dichotomiser 3)**. 
    El objetivo es predecir la especie de una flor Iris basándose en sus características morfológicas.
    """)
    st.image("imagenes/Infografia.png", caption="Flujo del Proyecto", use_column_width=True)
    
    st.subheader("Metodología CRISP-ML")
    st.info("""
    1. **Comprensión del Problema**: Clasificación multiclase de especies Iris.
    2. **Comprensión de Datos**: 150 instancias, 4 características numéricas.
    3. **Preparación**: División 80/20 y estratificación.
    4. **Modelado**: Árbol de decisión con criterio de Entropía.
    5. **Evaluación**: Análisis de precisión y matriz de confusión.
    6. **Despliegue**: Aplicación interactiva con Streamlit.
    """)

elif menu == "Exploración de Datos":
    st.header("Análisis Exploratorio de Datos (EDA)")
    st.write("Vista previa del dataset original:")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Estadísticas Descriptivas:")
        st.write(df.describe())
    with col2:
        st.write("Distribución de Clases:")
        fig, ax = plt.subplots()
        sns.countplot(x='class', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)

    st.subheader("Relación entre Variables")
    fig_pair = sns.pairplot(df, hue='class', palette='husl')
    st.pyplot(fig_pair)

elif menu == "Modelado y Evaluación":
    st.header("Entrenamiento y Resultados del Modelo")
    
    # Preparación simplificada para visualización
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    
    st.subheader("Visualización del Árbol de Decisión")
    fig_tree, ax_tree = plt.subplots(figsize=(20,10))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, ax=ax_tree)
    st.pyplot(fig_tree)
    
    st.subheader("Métricas de Rendimiento")
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    st.metric("Exactitud (Accuracy)", f"{acc*100:.2f}%")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("Matriz de Confusión:")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        st.pyplot(fig_cm)
    with col_m2:
        st.write("Importancia de las Variables:")
        importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='importance', y='feature', data=importances.sort_values('importance', ascending=False), ax=ax_imp)
        st.pyplot(fig_imp)

elif menu == "Predicción Interactiva":
    st.header("Realizar Nueva Predicción")
    st.write("Ajusta los parámetros de la flor para obtener la predicción del modelo:")
    
    # Re-entrenamiento rápido para asegurar estado
    X = df.drop('class', axis=1)
    y = df['class']
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X, y)
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
        sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    with col_in2:
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8)
        pw = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)
    
    input_data = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Resultado:")
        st.success(f"La especie predicha es: **{prediction}**")
    with res_col2:
        st.subheader("Probabilidades:")
        prob_df = pd.DataFrame(proba, columns=model.classes_)
        st.bar_chart(prob_df.T)
    
    st.subheader("Explicabilidad Local")
    if pw <= 0.8:
        st.info("💡 **Razonamiento**: Debido a que el ancho del pétalo es muy pequeño (≤ 0.8 cm), el modelo identifica esta flor con total certeza como **Iris-setosa**.")
    elif pw > 1.75:
        st.info("💡 **Razonamiento**: Con un ancho de pétalo superior a 1.75 cm, el modelo clasifica la muestra como **Iris-virginica**.")
    else:
        st.info(f"💡 **Razonamiento**: El ancho del pétalo ({pw}) está en el rango intermedio. El modelo utiliza la longitud del pétalo ({pl}) para decidir entre Versicolor y Virginica.")