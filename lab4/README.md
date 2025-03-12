# Proyecto de Análisis y Detección de Malware

## Universidad del Valle de Guatemala
- **Curso**: CC3094 – Security Data Science
- **Semestre**: I - 2025
- **Autores**:
  - Daniel Gómez (21429)
  - Adrian Rodríguez (21691)

## Descripción General

Este proyecto comprende una serie de herramientas para el análisis, clasificación y detección de malware utilizando técnicas de ciencia de datos y aprendizaje automático. El proyecto se divide en varias partes, cada una enfocada en diferentes aspectos del análisis de malware:

1. **Análisis de Familias de Malware**: Identificación y agrupación de malware en familias utilizando técnicas de clustering.
2. **Detección de Malware por Secuencias de APIs**: Implementación de modelos de Machine Learning y Deep Learning para la detección de malware basada en el comportamiento.

## Componentes Principales

### 1. Análisis de Familias de Malware

Esta parte del proyecto se enfoca en el análisis estático de archivos ejecutables para identificar familias de malware mediante técnicas de clustering.

#### Funcionalidades:
- **Desempaquetado de Malware**: Detección y descompresión recursiva de archivos ofuscados con UPX.
- **Extracción de Características PE**:
  - Metadatos del archivo (hash MD5, tamaño)
  - Información del encabezado PE
  - Detalles del encabezado opcional
  - Análisis de secciones (nombres, entropías, tamaños)
  - Bibliotecas y funciones importadas
  - Características de seguridad
- **Preprocesamiento de Datos**:
  - Creación de variables binarias para secciones y DLLs
  - Derivación de métricas adicionales
  - Codificación de variables categóricas
  - Imputación de valores faltantes
  - Escalado de características numéricas
  - Selección de características relevantes
- **Clustering y Análisis**:
  - Generación de embeddings con Gemini
  - Clustering con K-means (K=7)
  - Clustering con K-medoids (K=6)
  - Análisis de similitud con Índice de Jaccard
  - Visualización de grafos de similitud

### 2. Detección de Malware por Secuencias de APIs

Esta parte implementa dos enfoques distintos para la detección de malware utilizando secuencias de llamadas a APIs:

#### Modelo de Machine Learning (TF-IDF + Random Forest):
- Tokenización y normalización de secuencias de APIs
- Vectorización mediante TF-IDF (unigramas y bigramas)
- Clasificación con Random Forest (100 estimadores, profundidad máxima de 20)
- Alta precisión (98.86%) con mínimos falsos positivos
- Mayor interpretabilidad y eficiencia computacional

#### Modelo de Deep Learning (Embeddings de Gemini + Red Neuronal):
- Limpieza básica de texto
- Generación de embeddings mediante la API de Gemini (768 dimensiones)
- Red neuronal feedforward (capas de 64 y 32 neuronas con dropout)
- Alto recall (94.12%) para detectar más muestras de malware
- Mejor capacidad de generalización y estabilidad

## Requisitos

```
python 3.8+
pandas
numpy
scikit-learn
tensorflow
pefile
networkx
matplotlib
seaborn
upx
requests
```

## Instrucciones de Uso

### Configuración del Entorno

```bash
# Crear entorno virtual
python -m venv malware_env
source malware_env/bin/activate  # En Windows: malware_env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Análisis de Familias de Malware

```bash
# Ejecutar el procesamiento y extracción de características
python extract_features.py --malware_dir /path/to/malware/samples

# Generar y analizar clusters
python cluster_analysis.py --data_file data/processed_data.csv --api_key your_gemini_api_key
```

### Detección de Malware por Secuencias de APIs

```bash
# Ejecutar el modelo de Machine Learning
python ml_model.py --data_path path/to/api_sequences.csv

# Ejecutar el modelo de Deep Learning
python dl_model.py --data_path path/to/api_sequences.csv --api_key your_gemini_api_key

# Comparar resultados de ambos modelos
python compare_models.py --ml_results path/to/ml_results.json --dl_results path/to/dl_results.json
```

## Resultados y Comparación de Modelos

### Métricas en el Conjunto de Prueba

| Métrica             | Random Forest con TF-IDF | Red Neuronal con Embeddings |
|---------------------|--------------------------|------------------------------|
| Precisión (Accuracy)| 94.68%                  | 93.90%                       |
| Precisión (Precision)| 98.86%                 | 93.88%                       |
| Recuperación (Recall)| 90.39%                 | 94.12%                       |
| Puntuación F1       | 94.44%                  | 94.00%                       |
| AUC-ROC             | 98.29%                  | 97.75%                       |

### Análisis de Fortalezas y Debilidades

| Característica | Random Forest con TF-IDF | Red Neuronal con Embeddings |
|----------------|--------------------------|------------------------------|
| **Fortalezas** | - Alta precisión con mínimos falsos positivos<br>- Mayor interpretabilidad<br>- Menor dependencia de servicios externos<br>- Eficiencia computacional | - Alto recall (detecta más malware)<br>- Mayor estabilidad (menor varianza)<br>- Mejor generalización<br>- Captura relaciones semánticas complejas |
| **Debilidades** | - Menor recall (pierde algunas muestras de malware)<br>- Mayor variabilidad entre folds<br>- Adaptación limitada a nuevos patrones<br>- Preprocesamiento más complejo | - Mayor tasa de falsos positivos<br>- Menor interpretabilidad<br>- Dependencia de API externa<br>- Mayor complejidad de implementación |

## Contribución

Si desea contribuir a este proyecto, por favor envíe un pull request o abra un issue para discutir los cambios propuestos.

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

## Referencias

1. "Automated Behaviour-based Malware Detection Framework Based on NLP and Deep Learning Techniques"
2. Documentación de la API de Gemini para embeddings: https://ai.google.dev/gemini-api/docs/embeddings
3. Recursos adicionales sobre análisis de familias de malware y detección basada en comportamiento
