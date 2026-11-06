import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr

# 1. Definición de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "saferoute_finetuned.keras"

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "saferoute_completo.keras"

print("Iniciando SafeRoute App (Lógica Robusta Multi-Nivel)...")
print(f"Ruta del modelo: {MODEL_PATH}")

# 2. Cargar arquitectura y pesos del modelo
def load_model(weights_path):
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

    model = models.Model(inp, out)
    model.load_weights(str(weights_path))
    return model

model = load_model(MODEL_PATH)
print("¡Modelo cargado exitosamente!")

# 3. Mapeo de Clases Detalladas
CLASS_NAMES = [
    "c0: Conducción segura y atenta",
    "c1: Celular - Escribiendo (mano derecha)",
    "c2: Celular - Hablando (mano derecha)",
    "c3: Celular - Escribiendo (mano izquierda)",
    "c4: Celular - Hablando (mano izquierda)",
    "c5: Operando radio / tablero central",
    "c6: Tomando / Bebiendo líquido",
    "c7: Alcanzando objeto en asiento trasero",
    "c8: Acomodándose pelo o rostro",
    "c9: Desvío lateral / conversando"
]

def analyze_cabin_image(image):
    if image is None:
        return (
            "⚠️ Por favor ingresa una imagen o captura de cabina.",
            "Sin datos de telemetría.",
            {},
            {}
        )

    # Preprocesamiento MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Inferencia (10 probabilidades individuales)
    preds = model.predict(img_batch, verbose=0)[0]
    
    # 4. Agrupación Jerárquica de Riesgo Operativo (Resuelve ambigüedad izq/der)
    prob_seguro = float(preds[0])
    prob_celular = float(preds[1] + preds[2] + preds[3] + preds[4])
    prob_desvio_cabina = float(preds[5] + preds[6] + preds[7])
    prob_postura_rostro = float(preds[8] + preds[9])

    macro_confidences = {
        "🟢 Conducción Segura": prob_seguro,
        "🚨 Manipulación de Celular (Llamada/Texto)": prob_celular,
        "🟡 Operando Controles / Bebida / Asiento": prob_desvio_cabina,
        "🟠 Desatención Facial / Giro Lateral": prob_postura_rostro
    }

    # Diccionario detallado de las 10 clases para análisis secundario
    detailed_confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(10)}

    top_idx = int(np.argmax(preds))
    top_score = float(preds[top_idx]) * 100
    top_label = CLASS_NAMES[top_idx]

    # Lógica de decisión por Macro-Categoría de Riesgo
    if prob_celular > prob_seguro and prob_celular >= 0.35:
        alerta = f"🚨 RIESGO CRÍTICO: Manipulación de Celular Detectada ({prob_celular * 100:.1f}%)"
        telemetria = (
            f"⚠️ ALERTA EN CABINA: Predicción específica: '{top_label}'.\n"
            f"A 90 km/h en carretera, 3 segundos con la vista en el celular equivalen a recorrer ~75 metros a ciegas.\n"
            f"Acción preventiva: Activar micro-vibración háptica en volante y notificación visual periférica."
        )
    elif prob_seguro >= 0.40 and prob_seguro >= prob_celular and prob_seguro >= prob_desvio_cabina:
        alerta = f"🟢 ESTADO ÓPTIMO: Conducción Segura y Atenta ({prob_seguro * 100:.1f}%)"
        telemetria = (
            f"✅ Conductor en postura adecuada y atento a la ruta.\n"
            f"Tiempo de reacción estimado normal (~0.7s). Sumando kilometraje de conducción segura."
        )
    elif prob_desvio_cabina > prob_seguro:
        alerta = f"🟡 ADVERTENCIA MODERADA: Desvío hacia Controles / Cabina ({prob_desvio_cabina * 100:.1f}%)"
        telemetria = (
            f"ℹ️ Detección dominante: '{top_label}'.\n"
            f"Si la acción persiste por más de 2 segundos continuos, emitir chime sonoro suave.\n"
            f"Si la recurrencia es alta, sugerir Pausa Activa en la próxima estación de servicio."
        )
    else:
        alerta = f"🟠 ADVERTENCIA PREVENTIVA: Desatención o Postura Irregular ({max(prob_postura_rostro, 1.0 - prob_seguro) * 100:.1f}%)"
        telemetria = (
            f"ℹ️ Predicción tentativa: '{top_label}' ({top_score:.1f}%).\n"
            f"Postura corporal con desvío visual del camino. Mantener monitoreo pasivo."
        )

    return alerta, telemetria, macro_confidences, detailed_confidences

# 5. Interfaz de Gradio
with gr.Blocks(title="SafeRoute - Copiloto de Cabina", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚛 SafeRoute: Monitor Inteligente de Atención en Cabina")
    gr.Markdown(
        "Sistema de asistencia preventiva y no punitiva para conductores de transporte de carga y pasajeros en rutas de larga distancia. "
        "Integra un clasificador jerárquico basado en MobileNetV2 para prevenir siniestros viales por desatención."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(sources=["upload", "webcam"], type="pil", label="Captura de Cámara en Cabina")
            btn_predict = gr.Button("Analizar Fotograma", variant="primary")

        with gr.Column(scale=1):
            output_alert = gr.Textbox(label="Diagnóstico Preventivo SafeRoute", interactive=False)
            output_telemetry = gr.Textbox(label="Métrica de Impacto y Sugerencia de Asistencia", interactive=False, lines=4)
            output_macro = gr.Label(num_top_classes=4, label="Nivel de Riesgo Operativo Consolidado")
            with gr.Accordion("Ver desglose técnico por clase individual (10 clases)", open=False):
                output_detailed = gr.Label(num_top_classes=3, label="Top 3 Clases Específicas")

    btn_predict.click(
        fn=analyze_cabin_image,
        inputs=[input_img],
        outputs=[output_alert, output_telemetry, output_macro, output_detailed]
    )

if __name__ == "__main__":
    demo.launch(share=False)