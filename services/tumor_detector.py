import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

# Ruta del modelo
model_path = os.path.join(os.path.dirname(__file__), '..', 'brain_tumor_detector.h5')
model = load_model(model_path)

def analyze_brain_image(image_bytes: bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(image_array, cv.IMREAD_COLOR)

    # Preprocesamiento
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Redimensionar para el modelo
    image_resized = cv.resize(image, (240, 240)) / 255.0
    image_resized = image_resized.reshape((1, 240, 240, 3))

    result = model.predict(image_resized)
    tumor_detected = result > 0.5

    # Agregar texto sobre la imagen
    label = "Tumor detectado" if tumor_detected else "No se detectó tumor"
    color = (0, 0, 255) if tumor_detected else (0, 255, 0)
    font = cv.FONT_HERSHEY_SIMPLEX

    cv.putText(
        image, label, (10, 30), font, 1, color, 2, cv.LINE_AA
    )

    # 🔥 MAPA DE CALOR SIN AZUL
    thermal = cv.applyColorMap(gray, cv.COLORMAP_JET)

    # Eliminar zonas azules (componentes azules fuertes)
    mask_blue = thermal[:, :, 0] > 150  # Canal azul alto
    thermal[mask_blue] = [0, 0, 0]      # Eliminar azul: poner en negro

    # Mezclar imagen original con el térmico sin azul
    blended = cv.addWeighted(image, 0.6, thermal, 0.4, 0)

    # Guardar imagen temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv.imwrite(temp_file.name, blended)

    return bool(tumor_detected), temp_file.name
