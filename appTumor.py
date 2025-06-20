from flask import Flask, request, send_file, jsonify
from services.tumor_detector import analyze_brain_image
import os
import tempfile

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    file = request.files['file']
    image_bytes = file.read()

    modo = request.form.get('modo', 'tumor')

    if modo == 'tumor':
        detected, image_path = analyze_brain_image(image_bytes)

        if image_path and os.path.exists(image_path):
            response = send_file(image_path, mimetype='image/png')
            response.headers['Tumor-Detected'] = 'yes' if detected else 'no'
            return response
        else:
            return jsonify({'error': 'Error al procesar imagen'}), 500

    return jsonify({'error': 'Modo no válido'}), 400

if __name__ == '__main__':
    app.run(debug=True)