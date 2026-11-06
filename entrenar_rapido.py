import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Definir rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Detectar carpeta train de imágenes
TRAIN_DIR = RAW_DIR / "imgs" / "train"
if not TRAIN_DIR.exists():
    TRAIN_DIR = RAW_DIR / "train"

print(f"Directorio de imágenes: {TRAIN_DIR}")

# 2. Cargar splits procesados
df_train = pd.read_csv(PROCESSED_DIR / "train_split.csv")
df_val = pd.read_csv(PROCESSED_DIR / "val_split.csv")

# Normalizar nombres de columnas (eliminar espacios invisibles)
df_train.columns = df_train.columns.str.strip()
df_val.columns = df_val.columns.str.strip()

print(f"Columnas detectadas en train_split: {df_train.columns.tolist()}")

# Identificar columna de clase
posibles_clases = ['classname', 'class', 'label', 'target', 'category']
col_clase = next((c for c in posibles_clases if c in df_train.columns), None)
if not col_clase:
    col_clase = df_train.columns[1]

# Identificar columna de imagen
posibles_imgs = ['img', 'image', 'filepath', 'filename', 'path']
col_img = next((c for c in posibles_imgs if c in df_train.columns), None)
if not col_img:
    col_img = df_train.columns[0]

print(f"-> Usando como columna de imagen: '{col_img}'")
print(f"-> Usando como columna de clase:  '{col_clase}'")

# Asegurar formato string
df_train[col_clase] = df_train[col_clase].astype(str)
df_val[col_clase] = df_val[col_clase].astype(str)

# 3. Comprobar estructura de rutas en disco (subcarpeta c0/ o archivo plano)
ejemplo_img = str(df_train[col_img].iloc[0])
ruta_plana = TRAIN_DIR / ejemplo_img

if ruta_plana.exists():
    col_x_train = col_img
    col_x_val = col_img
else:
    # State Farm original: train/c0/img_xxx.jpg
    df_train['rel_path'] = df_train[col_clase] + '/' + df_train[col_img].astype(str)
    df_val['rel_path'] = df_val[col_clase] + '/' + df_val[col_img].astype(str)
    col_x_train = 'rel_path'
    col_x_val = 'rel_path'

# 4. Muestreo seguro sin alterar columnas (150 imágenes por clase para entrenar rápido)
def extraer_muestra(df, col_c, n_max):
    muestras = []
    for clase, grupo in df.groupby(col_c):
        muestras.append(grupo.sample(n=min(len(grupo), n_max), random_state=42))
    return pd.concat(muestras, ignore_index=True)

df_train_sub = extraer_muestra(df_train, col_clase, 150)
df_val_sub = extraer_muestra(df_val, col_clase, 30)

print(f"Total imágenes train muestra: {len(df_train_sub)}")
print(f"Total imágenes val muestra:   {len(df_val_sub)}")

# 5. Generadores con preprocesamiento MobileNetV2
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_gen = datagen.flow_from_dataframe(
    dataframe=df_train_sub,
    directory=str(TRAIN_DIR),
    x_col=col_x_train,
    y_col=col_clase,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df_val_sub,
    directory=str(TRAIN_DIR),
    x_col=col_x_val,
    y_col=col_clase,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# 6. Modelo base MobileNetV2
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inp = tf.keras.Input(shape=(224, 224, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
num_classes = len(train_gen.class_indices)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Entrenamiento de 2 épocas
print(f"\nIniciando entrenamiento ({num_classes} clases)...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2
)

# 8. Guardar modelo en models/
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
output_path = MODELS_DIR / "saferoute_base.keras"
model.save(output_path)

print(f"\n¡Éxito! Modelo guardado en: {output_path}")
print(f"Mapeo de clases: {train_gen.class_indices}")