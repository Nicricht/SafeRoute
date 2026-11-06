import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1. Definición de rutas y directorios
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = DATA_DIR / "raw" / "imgs" / "train"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INICIAL = MODELS_DIR / "saferoute_completo.keras"
if not MODEL_INICIAL.exists():
    MODEL_INICIAL = BASE_DIR / "saferoute_completo.keras"

MODEL_OUT = MODELS_DIR / "saferoute_finetuned.keras"

# Hiperparámetros de Fine-Tuning
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5  # Muy bajo para ajuste fino sin gradientes destructivos

print(f"Cargando splits desde: {PROCESSED_DIR}")
df_train = pd.read_csv(PROCESSED_DIR / "train_split.csv")
df_val = pd.read_csv(PROCESSED_DIR / "val_split.csv")

# Construir rutas absolutas de imagen
df_train['full_path'] = df_train.apply(lambda r: str(TRAIN_DIR / str(r['classname']) / str(r['img'])), axis=1)
df_val['full_path'] = df_val.apply(lambda r: str(TRAIN_DIR / str(r['classname']) / str(r['img'])), axis=1)

# Extraer etiqueta numérica (c0 -> 0, c1 -> 1, ...)
df_train['label'] = df_train['classname'].str.replace('c', '').astype(int)
df_val['label'] = df_val['classname'].str.replace('c', '').astype(int)

# Filtrar registros que realmente existan en disco
df_train = df_train[df_train['full_path'].apply(os.path.exists)]
df_val = df_val[df_val['full_path'].apply(os.path.exists)]

print(f"Imágenes válidas de entrenamiento: {len(df_train)}")
print(f"Imágenes válidas de validación (sujetos nuevos): {len(df_val)}")

# 2. Pipeline de Datos con Data Augmentation focalizado
def parse_image_train(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    # Data Augmentation leve para cabina: brillo, contraste y saturación
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def parse_image_val(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

ds_train = (
    tf.data.Dataset.from_tensor_slices((df_train['full_path'].values, df_train['label'].values))
    .shuffle(buffer_size=2048)
    .map(parse_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

ds_val = (
    tf.data.Dataset.from_tensor_slices((df_val['full_path'].values, df_val['label'].values))
    .map(parse_image_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# 3. Construir y cargar pesos actuales
print("Construyendo arquitectura base de MobileNetV2...")
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)
inp = tf.keras.Input(shape=(224, 224, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(10, activation="softmax")(x)

model = models.Model(inp, out)

print(f"Cargando pesos preentrenados desde: {MODEL_INICIAL}")
model.load_weights(str(MODEL_INICIAL))
print("¡Pesos cargados!")

# 4. Configurar el Descongelamiento (Unfreezing)
# Descongelamos las últimas 40 capas de MobileNetV2
base.trainable = True
capas_a_descongelar = 40
for layer in base.layers[:-capas_a_descongelar]:
    layer.trainable = False

entrenables = sum([1 for l in model.layers if l.trainable])
total_capas_base = len(base.layers)
print(f"Total capas en MobileNetV2: {total_capas_base}")
print(f"Capas descongeladas al final de la red: {capas_a_descongelar}")

# 5. Recompilar con Learning Rate bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Callbacks para entrenamiento controlado
callbacks = [
    ModelCheckpoint(
        filepath=str(MODEL_OUT),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# 7. Ejecución del Fine-Tuning
print("\n" + "="*60)
print("       INICIANDO FINE-TUNING DE SAFEROUTE")
print("="*60)

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n¡Fine-Tuning completado! El mejor modelo fue guardado en: {MODEL_OUT}")