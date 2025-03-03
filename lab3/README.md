# Laboratorio 3: Detección de Malware mediante Secuencias de Llamadas a APIs

## Universidad del Valle de Guatemala
- **Curso**: CC3094 – Security Data Science
- **Semestre**: I - 2025
- **Autores**:
  - Daniel Gómez 21429
  - Adrian Rodríguez 21691

## Descripción General

Este laboratorio implementa dos enfoques distintos para la detección de malware utilizando secuencias de llamadas a APIs:

1. **Modelo de Machine Learning (ML)**: Utiliza un clasificador Random Forest con características TF-IDF.
2. **Modelo de Deep Learning (DL)**: Emplea una red neuronal alimentada con embeddings generados por la API de Gemini.

El objetivo es comparar el rendimiento de ambos modelos y determinar cuál sería más adecuado para un sistema de detección de malware en producción.

## Dataset

El conjunto de datos contiene secuencias de llamadas a APIs extraídas de la ejecución de programas benignos y maliciosos, basado en el artículo "Automated Behaviour-based Malware Detection Framework Based on NLP and Deep Learning Techniques". Los datos se dividieron en:

- Conjunto de entrenamiento (70%)
- Conjunto de prueba (30%)

La división se realizó mediante muestreo estratificado para mantener la proporción de clases.

## Fases del Proyecto

### 1. Exploración y Preparación de Datos

En esta fase se realizó un análisis inicial del dataset para comprender:

- Distribución de clases (malware vs. benigno)
- Longitud de las secuencias de llamadas a APIs
- Frecuencia de aparición de distintas APIs
- Identificación de patrones comunes en ambas clases

Las secuencias de llamadas a APIs representan el comportamiento dinámico del software durante su ejecución, proporcionando información valiosa que puede evadir las técnicas de ofuscamiento que dificultan el análisis estático.

### 2. Preprocesamiento

#### Para el Modelo de Machine Learning:
- Tokenización de las secuencias de llamadas a APIs
- Eliminación de caracteres especiales y normalización
- Vectorización mediante TF-IDF, considerando tanto unigramas como bigramas
- Normalización de características para mejorar el rendimiento del modelo

#### Para el Modelo de Deep Learning:
- Limpieza básica de texto
- Preparación de secuencias para envío a la API de Gemini
- Generación de embeddings de dimensión 768 que capturan información semántica de las secuencias

### 3. Ingeniería de Características

#### Modelo ML:
- Se aplicó TF-IDF para convertir las secuencias en representaciones numéricas
- Se incluyeron unigramas y bigramas para capturar patrones de secuencia
- Se seleccionaron las características más relevantes mediante análisis de importancia

#### Modelo DL:
- Se utilizaron embeddings de Gemini como características de entrada
- Estos embeddings de 768 dimensiones capturan relaciones semánticas complejas entre las llamadas a APIs
- No se requirió ingeniería de características adicional gracias a la capacidad de los embeddings

### 4. Modelo 1: Random Forest con TF-IDF

Se implementó un clasificador Random Forest con las siguientes configuraciones:
- 100 estimadores
- Profundidad máxima de 20
- Criterio de división Gini
- Configuraciones adicionales optimizadas para este problema específico

Este modelo destaca por:
- Alta interpretabilidad
- Menor dependencia de servicios externos
- Eficiencia computacional en inferencia
- Excelente precisión con mínimos falsos positivos

### 5. Modelo 2: Red Neuronal con Embeddings de Gemini

Se diseñó una red neuronal feedforward con la siguiente arquitectura:
- Capa de entrada (768 dimensiones, correspondientes a los embeddings de Gemini)
- Primera capa densa (64 neuronas con activación ReLU)
- Capa de dropout (tasa 0.3)
- Segunda capa densa (32 neuronas con activación ReLU)
- Capa de dropout (tasa 0.2)
- Capa de salida (1 neurona con activación sigmoide)

La red se entrenó utilizando:
- Optimizador Adam
- Función de pérdida de entropía cruzada binaria
- Early stopping para prevenir el sobreajuste

Este modelo destaca por:
- Alto recall (detecta más muestras de malware)
- Mayor estabilidad entre diferentes ejecuciones
- Mejor generalización potencial
- Capacidad para capturar relaciones semánticas complejas

### 6. Evaluación y Validación

Para una evaluación robusta, se implementó:
- Validación cruzada de 10 folds
- Cálculo de múltiples métricas: accuracy, precision, recall, F1-score y AUC-ROC
- Matrices de confusión y curvas ROC para análisis detallado

### 7. Comparación de Modelos

#### Métricas en el Conjunto de Prueba

| Métrica             | Random Forest con TF-IDF | Red Neuronal con Embeddings |
|---------------------|--------------------------|------------------------------|
| Precisión (Accuracy)| 0.9468                  | 0.9390                       |
| Precisión (Precision)| 0.9886                 | 0.9388                       |
| Recuperación (Recall)| 0.9039                 | 0.9412                       |
| Puntuación F1       | 0.9444                  | 0.9400                       |
| AUC-ROC             | 0.9829                  | 0.9775                       |

#### Resultados de Validación Cruzada (10-folds)

| Métrica             | Random Forest con TF-IDF      | Red Neuronal con Embeddings   |
|---------------------|-------------------------------|-------------------------------|
| Precisión (Accuracy)| 0.9355 (±0.0257)             | 0.9500 (±0.0139)              |
| Precisión (Precision)| 0.9839 (±0.0170)            | 0.9540 (±0.0199)              |
| Recuperación (Recall)| 0.8856 (±0.0445)            | 0.9452 (±0.0146)              |
| Puntuación F1       | 0.9317 (±0.0281)             | 0.9494 (±0.0140)              |
| AUC-ROC             | 0.9849 (±0.0092)             | 0.9834 (±0.0088)              |

#### Análisis de Fortalezas y Debilidades

| Característica       | Random Forest con TF-IDF | Red Neuronal con Embeddings |
|----------------------|--------------------------|------------------------------|
| **Fortalezas**       | - Alta precisión con mínimos falsos positivos<br>- Mayor interpretabilidad<br>- Menor dependencia de servicios externos<br>- Eficiencia computacional | - Alto recall (detecta más malware)<br>- Mayor estabilidad (menor varianza)<br>- Mejor generalización<br>- Captura relaciones semánticas complejas |
| **Debilidades**      | - Menor recall (pierde algunas muestras de malware)<br>- Mayor variabilidad entre folds<br>- Adaptación limitada a nuevos patrones<br>- Preprocesamiento más complejo | - Mayor tasa de falsos positivos<br>- Menor interpretabilidad<br>- Dependencia de API externa<br>- Mayor complejidad de implementación |

### 8. Conclusiones

1. Ambos modelos muestran un rendimiento excepcional en la detección de malware basada en secuencias de llamadas a APIs, con métricas generales por encima del 93% en todas las dimensiones evaluadas.

2. El modelo de Random Forest con TF-IDF destaca por su alta precisión y baja tasa de falsos positivos, haciéndolo particularmente adecuado para sistemas antivirus destinados a usuarios finales donde las interrupciones causadas por falsos positivos deben minimizarse.

3. El modelo de red neuronal con embeddings de Gemini muestra un mayor recall y estabilidad, sugiriendo una mejor capacidad para detectar una gama más amplia de malware, incluyendo potencialmente variantes menos comunes o más nuevas.

4. La elección entre ambos modelos depende fundamentalmente del contexto de implementación y del equilibrio deseado entre minimizar falsos positivos y maximizar la detección de amenazas.

5. Los embeddings generados por la API de Gemini demuestran ser una representación poderosa para capturar relaciones semánticas en secuencias de llamadas a APIs, ofreciendo una alternativa prometedora a enfoques tradicionales como TF-IDF.

6. La interpretabilidad del modelo Random Forest proporciona ventajas significativas para el análisis forense y la comprensión de los patrones de comportamiento del malware.

## Instrucciones de Uso

1. **Configuración del Entorno**:
   ```
   # Crear entorno virtual
   python -m venv malware_env
   source malware_env/bin/activate  # En Windows: malware_env\Scripts\activate
   
   # Instalar dependencias
   pip install -r requirements.txt
   ```

2. **Ejecución del Modelo ML**:
   ```
   python ml_model.py --data_path path/to/api_sequences.csv
   ```

3. **Ejecución del Modelo DL**:
   ```
   python dl_model.py --data_path path/to/api_sequences.csv --api_key your_gemini_api_key
   ```

4. **Evaluación y Comparación**:
   ```
   python compare_models.py --ml_results path/to/ml_results.json --dl_results path/to/dl_results.json
   ```

## Referencias

1. "Automated Behaviour-based Malware Detection Framework Based on NLP and Deep Learning Techniques" - Referencia del artículo base para la construcción del dataset.
2. Documentación de la API de Gemini para embeddings de clasificación: https://ai.google.dev/gemini-api/docs/embeddings
3. Documentación sobre tipos de tareas para embeddings de Gemini: https://ai.google.dev/api/embeddings#v1beta.TaskType
4. Tutorial sobre clasificadores de texto con embeddings: https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/text_classifier_embeddings.ipynb
5. Recursos adicionales sobre embeddings y puntuaciones de similitud: https://www.kaggle.com/code/markishere/day-2-embeddings-and-similarity-scores
