import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from deepface import DeepFace

# DeepFace detecta automáticamente las emociones faciales
def predict_emotion_from_face(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)
        return result[0]['dominant_emotion'].capitalize()
    except Exception as e:
        print("❌ Error en DeepFace:", e)
        return "No detectado"

def analyze_emotion_image(image_bytes: bytes):
    mp_face_mesh = mp.solutions.face_mesh

    # Decodificar imagen desde bytes
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return False, None

    # Redimensionar y convertir para MediaPipe
    image = cv2.resize(image, (500, 500))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # Puntos clave que queremos marcar
    puntos_deseados = [468, 473, 133, 33, 362, 263, 55, 70, 285, 300, 4, 185, 306, 0, 17]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.line(image_gray_bgr, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                        cv2.line(image_gray_bgr, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)

    # Guardar imagen temporal para analizarla con DeepFace
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, image)

    # Detectar emoción real
    emotion = predict_emotion_from_face(temp_file.name)

    # Escribir emoción en la imagen final
    cv2.putText(image_gray_bgr, f"Emocion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Sobrescribir la imagen final procesada
    cv2.imwrite(temp_file.name, image_gray_bgr)
    return True, temp_file.name
