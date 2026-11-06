import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1. Definición de rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = RAW_DIR / "imgs" / "train"
if not TRAIN_DIR.exists():
    TRAIN_DIR = RAW_DIR / "train"

print(f"-> Directorio de imágenes: {TRAIN_DIR}")

# 2. Cargar datos completos
df_train = pd.read_csv(PROCESSED_DIR / "train_split.csv")
df_val = pd.read_csv(PROCESSED_DIR / "val_split.csv")

# Limpieza de nombres de columnas
df_train.columns = df_train.columns.str.strip()
df_val.columns = df_val.columns.str.strip()

df_train['classname'] = df_train['classname'].astype(str)
df_val['classname'] = df_val['classname'].astype(str)

# Construir ruta relativa según estructura de carpetas
ejemplo_path = TRAIN_DIR / str(df_train['img'].iloc[0])
if ejemplo_path.exists():
    x_col = 'img'
else:
    df_train['rel_path'] = df_train['classname'] + '/' + df_train['img'].astype(str)
    df_val['rel_path'] = df_val['classname'] + '/' + df_val['img'].astype(str)
    x_col = 'rel_path'

print(f"-> Total imágenes entrenamiento (completo): {len(df_train)}")
print(f"-> Total imágenes validación (completo):    {len(df_val)}")

# 3. Data Augmentation para regularización
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False  # No voltear para no confundir mano izquierda con derecha
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=str(TRAIN_DIR),
    x_col=x_col,
    y_col='classname',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=df_val,
    directory=str(TRAIN_DIR),
    x_col=x_col,
    y_col='classname',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 4. Arquitectura con MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks
best_model_path = str(MODELS_DIR / "saferoute_completo.keras")
best_weights_path = str(MODELS_DIR / "saferoute_weights.weights.h5")

callbacks = [
    ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]

# 6. Ciclo de Entrenamiento
EPOCHS = 10
print("\n" + "="*50)
print(f"Iniciando entrenamiento completo del modelo SafeRoute ({EPOCHS} épocas)...")
print("="*50)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 7. Guardar pesos en formato HDF5
model.save_weights(best_weights_path)
print("\n" + "="*50)
print(f"-> Modelo completo guardado en: {best_model_path}")
print(f"-> Pesos exportados listos en: {best_weights_path}")
print("="*50)