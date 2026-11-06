import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# 1. Definición de rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TEST_CSV = PROCESSED_DIR / "test_split.csv"
DATA_RAW = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "saferoute_finetuned.keras"

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "saferoute_completo.keras"

TRAIN_DIR = DATA_RAW / "imgs" / "train"

CLASS_NAMES = [
    "c0: Seguro",
    "c1: Texto Der",
    "c2: Llamada Der",
    "c3: Texto Izq",
    "c4: Llamada Izq",
    "c5: Radio",
    "c6: Bebiendo",
    "c7: Alcanzar atrás",
    "c8: Pelo/Maquillaje",
    "c9: Copiloto"
]

def load_saferoute_model(weights_path):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base.trainable = False
    inp = tf.keras.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(10, activation="softmax")(x)
    
    model = Model(inp, out)
    model.load_weights(str(weights_path))
    return model

def load_and_fix_csv(csv_path):
    """Carga el CSV asegurando que 'subject', 'classname' e 'img' sean 3 columnas independientes."""
    df = pd.read_csv(csv_path, sep=None, engine='python')
    
    # Si se cargó todo en una sola columna por temas de delimitador
    if len(df.columns) == 1:
        col_name = df.columns[0]
        split_cols = df[col_name].str.split(',', expand=True)
        split_cols.columns = [c.strip() for c in col_name.split(',')]
        df = split_cols
        
    # Limpiar posibles espacios en blanco
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        
    return df

def evaluate_test_set():
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"No existe el archivo {TEST_CSV}")

    df_test = load_and_fix_csv(TEST_CSV)
    print(f"Total imágenes en test: {len(df_test)}")
    print(f"Columnas detectadas: {list(df_test.columns)}")
    print("Muestra de datos:")
    print(df_test.head(2))

    # Prueba de existencia de la primera imagen
    sample_cls = df_test.iloc[0]['classname']
    sample_img = df_test.iloc[0]['img']
    sample_path = TRAIN_DIR / sample_cls / sample_img

    print(f"\nProbando ruta de muestra: {sample_path}")
    if not sample_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {sample_path}\n"
            f"Verifica si dentro de '{TRAIN_DIR}' existe la subcarpeta '{sample_cls}'."
        )
    print("¡Ruta de imagen validada correctamente en disco!")

    print("\nCargando arquitectura y pesos del modelo...")
    model = load_saferoute_model(MODEL_PATH)
    print("Modelo cargado con éxito.")

    y_true = []
    y_pred = []
    missing_count = 0

    print("Evaluando imágenes en conductores no vistos...")
    for idx, row in df_test.iterrows():
        img_path = TRAIN_DIR / row['classname'] / row['img']

        if not img_path.exists():
            missing_count += 1
            continue

        with Image.open(img_path) as raw_img:
            img = raw_img.convert('RGB').resize((224, 224))
            img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32))
            img_batch = np.expand_dims(img_arr, axis=0)

        preds = model.predict(img_batch, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        true_class = int(row['classname'].replace('c', ''))

        y_true.append(true_class)
        y_pred.append(pred_class)

        if len(y_true) % 250 == 0:
            print(f"Evaluadas {len(y_true)}/{len(df_test)} imágenes...")

    if missing_count > 0:
        print(f"\nAviso: {missing_count} imágenes no se pudieron abrir.")

    print(f"\nEvaluación finalizada sobre {len(y_true)} imágenes.")

    # 1. Reporte de Clasificación (Precision, Recall, F1-Score)
    print("\n" + "="*60)
    print("        REPORTE DE CLASIFICACIÓN FINAL (TEST SET)")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 2. Matriz de Confusión Normalizada
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Matriz de Confusión Normalizada - SafeRoute (MobileNetV2)", fontsize=13, weight='bold')
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_path = MODELS_DIR / "matriz_confusion.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nMatriz de confusión guardada en: {plot_path}")

if __name__ == "__main__":
    evaluate_test_set()