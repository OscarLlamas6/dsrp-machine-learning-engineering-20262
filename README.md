# Ingeniería de Machine Learning Avanzado

![Machine Learning Engineering](module1-feature-engineering/notebooks/assets/header.png)

Un curso práctico de cuatro módulos que te lleva desde la **ingeniería de
características (feature engineering)** hasta los **sistemas de ML en producción** y
las **aplicaciones modernas de GenAI**. Cada módulo combina la *teoría y la
matemática* con *código ejecutable*, *datasets reales* y las *herramientas de
MLOps* que se usan en la industria (Feast, MLflow, Qdrant, Docker, Airflow,
LangGraph, MCP).

> **Para quién es:** científicos de datos / ingenieros de ML que ya conocen lo
> básico de Python, NumPy, pandas y scikit-learn y quieren dar el salto a la parte
> de ingeniería e investigación del ML.

---

## 🎯 Resultados de aprendizaje

Al terminar el curso serás capaz de:

- Construir características de la forma correcta y servirlas a través de un
  **feature store** (Feast), orquestando el pipeline con **Apache Airflow**.
- Entender y aplicar **regularización**, **ensembles**, **SVM**, **aprendizaje no
  supervisado** y **redes neuronales** — y rastrear cada experimento con **MLflow**.
- Modelar y pronosticar **series de tiempo** con enfoques clásicos y de ML.
- Construir una aplicación completa de **RAG** sobre una **base de datos
  vectorial** (Qdrant) con una interfaz de chat en **Streamlit**, diseñar
  **agentes de IA** con **LangGraph/LangChain** y exponer herramientas mediante el
  **Model Context Protocol (MCP)**.

---

## 🗂️ Estructura del repositorio

```
.
├── README.md                         ← estás aquí
├── module1-feature-engineering/      ← Feature engineering + feature store Feast + Airflow
├── module2-advanced-ml/              ← Regularización, ensembles, SVM, no supervisado, redes neuronales + MLflow
├── module3-time-series/              ← Descomposición, ARIMA, XGBoost, clustering de series
└── module4-genai/                    ← Bases de datos vectoriales, RAG, agentes, MCP
```

Cada módulo es autocontenido: trae su propio `README.md`, sus dependencias,
notebooks y (cuando aplica) un `docker-compose.yml` para la infraestructura de
soporte.

---

## 📦 Módulo 1 — Feature Engineering & Feature Stores

| Tema | Qué aprenderás |
|------|----------------|
| Selección de variables | Filtro (varianza, MI, chi²), wrapper (RFE), embebido (L1, importancia de árboles); selección vs. extracción |
| Imputación | Valores faltantes: media/mediana/moda, KNN, variable indicadora |
| Encoding | Label/ordinal, one-hot, feature hashing (y sus trampas) |
| Transformaciones | Log/log1p, Box-Cox y Yeo-Johnson, bucketing / binning |
| Escalado | Z-score, robusto (mediana/IQR), min-max y normalización L2 |
| Reducción de dimensionalidad | PCA (matemática + intuición) |
| Embeddings | Representaciones densas aprendidas para categóricas/texto |
| **Detección de outliers / anomalías** | Estadísticos (z-score, IQR/Tukey, MAD) y basados en ML (Isolation Forest, DBSCAN); eliminar / winsorizar / transformar |
| **Feature stores** | Concepto, serving online/offline, correctitud point-in-time con **Feast** |
| **Pipeline de entrenamiento** | Cerrar el ciclo: feature store → set de entrenamiento → modelo entrenado y evaluado (el experiment tracking con MLflow llega en el **Módulo 2**) |
| **Orquestación** | El orquestador **Apache Airflow** compartido del curso (introducido aquí) corriendo todo el pipeline |

**Entregables**
- `notebooks/01_feature_engineering_theory.ipynb` — selección de variables,
  imputación, encoding (label/one-hot/hashing), transformaciones (log,
  Box-Cox/Yeo-Johnson, binning), escalado, PCA y embeddings, con matemática y
  ejemplos sobre un dataset real (en español).
- `notebooks/02_outlier_detection.ipynb` — detección de outliers / anomalías como
  paso de limpieza: métodos estadísticos (z-score, IQR/Tukey, MAD) y basados en ML
  (Isolation Forest, DBSCAN), más cómo tratarlos (en español).
- `notebooks/03_feature_pipeline_feast.ipynb` — un pipeline de features que
  materializa y sirve las características (en español).
- `notebooks/04_pipeline_entrenamiento.ipynb` — cierra el ciclo: construye un set
  de entrenamiento desde Feast, entrena y evalúa un modelo (en español). El
  experiment tracking con MLflow se introduce en el Módulo 2.
- `platform/` — la **plataforma compartida del curso**: un único
  `docker-compose.yml` con Feast (Redis + Postgres) y el stack de **Airflow**
  compartido. Contiene `feature_repo/` (Feast) y `dags/` (el DAG del Módulo 1:
  extract → prepare → transform → validate → feast_apply → feast_materialize →
  train_model). El experiment tracking (MLflow) se agrega en el Módulo 2.

## 📦 Módulo 2 — Algoritmos de ML Avanzado (+ MLflow)

| Tema | Algoritmos |
|------|-----------|
| Regularización | Ridge (L2), Lasso (L1), ElasticNet — la historia de sesgo/varianza |
| Ensembles | Bagging y boosting → Random Forest, **XGBoost**, **CatBoost**, **LightGBM** (y en qué se diferencian); voting y stacking |
| Métodos de margen | Máquinas de Soporte Vectorial (kernels, el dual) |
| No supervisado | DBSCAN, OPTICS, t-SNE, UMAP, modelos de mezcla gaussiana |
| Redes neuronales | Optimización, backpropagation, activaciones, regularización, hiperparámetros — en **PyTorch** (regresión y clasificación) |

Cada notebook registra runs, parámetros, métricas y artefactos en **MLflow** y
registra el mejor modelo en el **Model Registry de MLflow**.

## 📦 Módulo 3 — Series de Tiempo

| Tema | Qué aprenderás |
|------|----------------|
| Descomposición | Tendencia / estacionalidad / residuo (aditiva y multiplicativa, STL) |
| ARIMA | Estacionariedad, ACF/PACF, modelado (S)ARIMA y diagnósticos |
| ML para series | Forecasting basado en features con **XGBoost** (rezagos, ventanas, features de calendario) |
| Clustering de series | DTW, clustering por forma de varias series |

## 📦 Módulo 4 — Bases de Datos Vectoriales, RAG, Agentes & MCP

| Tema | Qué construirás |
|------|-----------------|
| Embeddings y recuperación | Embeddings densos, búsqueda por similitud |
| Chunking | Estrategias y compromisos |
| Búsqueda híbrida | Fusión densa + dispersa (BM25) |
| Base vectorial | Una instancia de **Qdrant** dockerizada |
| RAG | Un pipeline de punta a punta con un chat en **Streamlit** |
| Agentes | Un agente **LangGraph/LangChain** con flujo de control explícito |
| MCP | Un **servidor MCP** + **cliente** con el SDK `mcp` de Python |

---

## 🚀 Cómo empezar

```bash
# 1. Instala las dependencias del Módulo 1 con uv (crea .venv desde pyproject.toml + uv.lock)
cd module1-feature-engineering && uv sync

# 2. Levanta la plataforma compartida (Feast + Airflow del curso; MLflow llega en el Módulo 2)
cd platform && docker compose up -d --build

# 3. Abre Jupyter
cd .. && uv run jupyter lab
```

> **Tip:** el `README.md` de cada módulo tiene los comandos exactos para sus
> servicios de Docker (Feast, MLflow, Qdrant) y cómo correr sus notebooks/apps.

## 🧰 Prerrequisitos

- Python 3.10+ (se recomienda 3.11)
- [**uv**](https://docs.astral.sh/uv/) para gestionar dependencias
- Docker y Docker Compose
- ~8 GB de RAM libre para los notebooks más pesados (UMAP, PyTorch, XGBoost)
- Una API key de un LLM para el RAG/agentes del Módulo 4 (p. ej.
  `ANTHROPIC_API_KEY`); se documentan modelos locales como alternativa.

---

## 📚 Cronograma sugerido (8 semanas)

| Semanas | Módulo |
|---------|--------|
| 1–2 | Módulo 1 — Feature Engineering & Feast |
| 3–5 | Módulo 2 — ML Avanzado + MLflow |
| 6 | Módulo 3 — Series de Tiempo |
| 7–8 | Módulo 4 — Bases vectoriales, RAG, Agentes, MCP |

## 📄 Licencia

Material del curso publicado para uso educativo. Revisa las licencias de cada
librería de terceros.
