# Proyecto de Clasificación Iris con Árbol de Decisión ID3

## Descripción del Proyecto

Este proyecto implementa un clasificador de árbol de decisión utilizando el algoritmo ID3 (basado en entropía) para predecir la especie de flores Iris basado en cuatro características morfológicas: largo y ancho del sépalo, y largo y ancho del pétalo.

El dataset Iris contiene 150 instancias distribuidas equitativamente entre tres especies:
- Iris-setosa
- Iris-versicolor  
- Iris-virginica

## Tecnologías Utilizadas

- **Python 3.x**: Lenguaje de programación principal
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib & Seaborn**: Visualización de datos
- **Scikit-learn**: Implementación del algoritmo de árbol de decisión
- **Streamlit**: Aplicación web interactiva (para app.py)
- **Jupyter Notebook**: Desarrollo y experimentación inicial

## Estructura del Proyecto

```
ML_Project_Builder_Output/
│
├── README.md           # Este archivo
├── Tutorial.md         # Guía paso a paso para reproducir el proyecto
├── app.py              # Aplicación Streamlit interactiva
├── technical_documentation.md  # Documentación técnica completa
├── landing_page/       # Landing page profesional (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── script.js
└── imagenes/           # Recursos visuales del proyecto
    ├── Arbol.png
    ├── Grafica_de_barras.png
    ├── Infografia.png
    └── Matriz_de_confusion.png
```

## Métricas de Rendimiento

El modelo alcanzó un rendimiento excelente:
- **Exactitud (Accuracy)**: 96.67%
- **Precisión por clase**:
  - Iris-setosa: 100%
  - Iris-versicolor: 100%
  - Iris-virginica: 86%
- **Recall por clase**:
  - Iris-setosa: 100%
  - Iris-versicolor: 92%
  - Iris-virginica: 100%

## Requisitos de Instalación

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

## Instrucciones de Ejecución

1. **Ejecutar el notebook original**:
   ```
   jupyter notebook Arbol_ID3_Estudiante_Iris.ipynb
   ```

2. **Ejecutar la aplicación Streamlit**:
   ```
   streamlit run app.py
   ```

3. **Ver la landing page**:
   Abrir `landing_page/index.html` en cualquier navegador web

## Características Principales

- ✅ Implementación completa del algoritmo ID3 usando entropía como criterio
- ✅ Visualización detallada del árbol de decisión entrenado
- ✅ Evaluación comprehensiva con matriz de confusión y reporte de clasificación
- ✅ Aplicación interactiva para hacer predicciones en tiempo real
- ✅ Documentación técnica completa siguiendo estándares académicos
- ✅ Landing page profesional con explicación del proyecto
- ✅ Tutorial paso a paso para reproducir todo el proceso

## Arquitectura del Modelo

El árbol de decisión aprendió las siguientes reglas de clasificación:

1. **Regla Setosa**: Si el ancho del pétalo ≤ 0.8 cm → Iris-setosa
2. **Regla Virginica**: Si el ancho del pétalo > 1.75 cm → Iris-virginica  
3. **Regla Versicolor**: Si 0.8 cm < ancho del pétalo ≤ 1.75 cm Y longitud del pétalo ≤ 4.95 cm → Iris-versicolor

## Contribuciones

Este proyecto demuestra:
- Aplicación práctica de algoritmos de machine learning clásicos
- Buenas prácticas en documentación de proyectos de datos
- Creación de aplicaciones interactivas para demostración de modelos
- Enfoque profesional en la presentación de trabajo científico

---
*Proyecto desarrollado como parte del seguimiento 3 - Especialización en Machine Learning*