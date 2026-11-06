from pathlib import Path
import pandas as pd
import numpy as np

# Subir a la raíz del proyecto SafeRoute
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Buscar el archivo de metadatos
posibles_rutas = [
    DATA_RAW / "imgs" / "driver_imgs_list.csv",
    DATA_RAW / "driver_imgs_list.csv"
]

csv_path = next((p for p in posibles_rutas if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(f"No se encontró driver_imgs_list.csv en {posibles_rutas}")

print(f"Cargando dataset desde: {csv_path}")
df = pd.read_csv(csv_path)

# Limpiar nombres de columnas y espacios
df.columns = [c.strip() for c in df.columns]
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

subjects = df['subject'].unique()
total_subs = len(subjects)
print(f"Total imágenes detectadas: {len(df)}")
print(f"Total conductores identificados: {total_subs}")

# Fijar semilla para reproducibilidad
np.random.seed(42)
shuffled_subjects = np.random.permutation(subjects)

# Partición proporcional dinámica (70% train / 15% val / 15% test)
n_train = int(total_subs * 0.70)  # 14 sujetos
n_val = int(total_subs * 0.15)    # 3 sujetos

train_subs = shuffled_subjects[:n_train]
val_subs = shuffled_subjects[n_train:n_train + n_val]
test_subs = shuffled_subjects[n_train + n_val:]  # Restantes (3 sujetos)

df_train = df[df['subject'].isin(train_subs)]
df_val = df[df['subject'].isin(val_subs)]
df_test = df[df['subject'].isin(test_subs)]

print("\n--- DISTRIBUCIÓN POR CONJUNTO ---")
print(f"Train: {len(train_subs)} sujetos -> {len(df_train)} imágenes ({len(df_train)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_subs)} sujetos -> {len(df_val)} imágenes ({len(df_val)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_subs)} sujetos -> {len(df_test)} imágenes ({len(df_test)/len(df)*100:.1f}%)")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df_train.to_csv(PROCESSED_DIR / "train_split.csv", index=False)
df_val.to_csv(PROCESSED_DIR / "val_split.csv", index=False)
df_test.to_csv(PROCESSED_DIR / "test_split.csv", index=False)

print("\n¡Archivos train_split.csv, val_split.csv y test_split.csv regenerados con éxito!")