# 🚛 SafeRoute: Sistema Inteligente de Detección Temprana de Distracciones en Cabina

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://www.tensorflow.org/)
[![UI](https://img.shields.io/badge/Interface-Gradio-red.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/TRL-Nivel%204--5%20(Prototipo)-blueviolet.svg)]()

> **Proyecto Capstone para el Samsung Innovation Campus (SIC) - Curso de Inteligencia Artificial.**  
> Solución basada en visión por computadora y aprendizaje profundo orientada a la asistencia preventiva y reducción de siniestros viales en el sector del transporte profesional de carga y pasajeros.

---

## 📌 Contexto y Definición del Problema

En el transporte de carga interurbano y de pasajeros, las extensas jornadas de conducción y la monotonía de la ruta generan altos niveles de fatiga y propensión a distracciones manuales y visuales. A velocidades de autopista (90–100 km/h), apartar la mirada del camino durante 2 a 3 segundos equivale a recorrer cerca de 80 metros a ciegas.

### Pregunta Guía (Design Thinking)
> *"¿Cómo podríamos alertar oportunamente a los transportistas de carga en rutas interurbanas cuando se distraen (uso de teléfono, bebidas o desvío de atención), sin generar alarmas invasivas que aumenten el estrés en cabina, para prevenir colisiones graves causadas por la monotonía de la ruta y las extensas jornadas?"*

---

## 🎯 Objetivos del Proyecto

1. **Clasificación de Comportamiento:** Entrenar y optimizar una red neuronal convolucional (CNN) con *Transfer Learning* capaz de clasificar 10 estados de conducción a partir de imágenes de cabina.
2. **Generalización Robusta:** Implementar una estrategia de partición de datos agrupada por conductor (*Subject Split*) para evitar la fuga de información (*Data Leakage*) y evaluar el modelo en condiciones reales no memorizadas.
3. **Prototipo Funcional (TRL 4–5):** Desplegar una aplicación web interactiva en **Gradio** que permita cargar imágenes o transmitir vídeo para inferir la postura en tiempo real y emitir alertas preventivas inmediatas ante conductas de riesgo.

---

## 📊 Dataset Utilizado

El proyecto utiliza el dataset público **State Farm Distracted Driver Detection** (Kaggle), compuesto por **22.424 imágenes etiquetadas** en resolución $640 \times 480$ píxeles distribuidas en 10 categorías:

| Código | Descripción oficial | Impacto en cabina |
| :---: | :--- | :--- |
| `c0` | **Conducción segura y atenta** | Estado óptimo |
| `c1` | Escribiendo texto (mano derecha) | Riesgo crítico (visual y manual) |
| `c2` | Hablando por teléfono (mano derecha) | Riesgo alto (cognitivo y manual) |
| `c3` | Escribiendo texto (mano izquierda) | Riesgo crítico (visual y manual) |
| `c4` | Hablando por teléfono (mano izquierda) | Riesgo alto (cognitivo y manual) |
| `c5` | Operando la radio / tablero | Riesgo moderado (visual momentáneo) |
| `c6` | Bebiendo líquidos | Riesgo moderado (manual) |
| `c7` | Alcanzando objeto en asiento trasero | Riesgo crítico (desvío total de vista) |
| `c8` | Arreglándose el pelo / maquillaje | Riesgo alto (desvío de atención) |
| `c9` | Hablando con acompañante / desvío lateral | Riesgo moderado (desatención de ruta) |

---

## 🛠️ Arquitectura y Metodología (CRISP-DM)

```text
 ┌──────────────────────┐
 │   Datos de Cabina    │
 │ (State Farm Dataset) │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐      • Redimensionamiento (224x224)
 │   Preprocesamiento   │ ───► • Data Augmentation (rotación, brillo, zoom)
 │   & Split Sujetos    │      • División 70% Train / 15% Val / 15% Test
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐      • Baseline: CNN personalizada de 3 capas
 │  Modelado Predictivo │ ───► • Transfer Learning: MobileNetV2 (ImageNet)
 │  (Deep Learning)     │      • Fine-Tuning y optimización con Adam
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐      • Métricas: Accuracy, Precision, Recall, F1-Score
 │ Evaluación & Métricas│ ───► • Matriz de Confusión por sujeto no visto
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Despliegue en Gradio │ ───► Interfaz interactiva de asistencia en tiempo real
 └──────────────────────┘
```

---

## 📂 Estructura del Repositorio

```text
SafeRoute/
├── data/
│   ├── raw/                  # Imágenes originales del dataset 
│   ├── processed/            # Divisiones train_split.csv, val_split.csv, test_split.csv
│   └── driver_imgs_list.csv  # Metadatos de asignación de sujetos
├── notebooks/
│   ├── 01_exploracion_datos.ipynb    # EDA, balance de clases y partición por sujetos
│   ├── 02_preprocesamiento.ipynb     # Pipelines y Data Augmentation
│   ├── 03_modelo_baseline.ipynb      # Modelo convolucional base
│   └── 04_transfer_learning.ipynb    # MobileNetV2, entrenamiento y benchmarking
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Generadores y transformaciones de datos
│   ├── model.py              # Definición de arquitecturas
│   ├── train.py              # Ciclo de entrenamiento
│   └── evaluate.py           # Cálculo de matrices de confusión y métricas
├── models/
│   └── .gitkeep              # Directorio para almacenar pesos (.keras)
├── app/
│   └── app.py                # Aplicación interactiva en Gradio
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Nicricht/SafeRoute.git
cd SafeRoute
```

### 2. Crear y activar el entorno virtual
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación Prototipo
```bash
python app/app.py
```
Abre en tu navegador la dirección local generada: `http://127.0.0.1:7860`.

---

## 👥 Equipo de Trabajo

| Integrante | Rol Principal | Responsabilidades |
| :--- | :--- | :--- |
| **Eduardo Vega** |  |  |
| **Nicolás Vega** |  |  |
| **Juan Velásquez** |  |  |

---

## 📜 Licencia y Agradecimientos

Proyecto desarrollado en el marco del programa **Samsung Innovation Campus 2026** impartido por **Innovacien**.