from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import os
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelos.preProcess import preProcess
from modelos.oneClass import load_images_from_folder, scaler, mean_normal_image, thresh_anomaly_score, max_anomaly_score
app = Flask(__name__)

cmap = LinearSegmentedColormap.from_list("anomaly_cmap", ["green", "yellow", "red"])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_data = base64.b64decode(data)
    nparr = np.frombuffer(img_data, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    temp_input = "temp_input.png"
    temp_output = "temp_output.png"

    cv2.imwrite(temp_input, original_image)
    preProcess(temp_input, "./", debug=False)
    preprocessed_image = cv2.imread(temp_output, cv2.IMREAD_GRAYSCALE)

    resized_image = cv2.resize(preprocessed_image, (256, 256))
    test_image_flat = scaler.transform(resized_image.flatten().reshape(1, -1))

    diff_image = np.abs(test_image_flat.reshape(256, 256) - mean_normal_image)
    anomaly_score = float(np.mean(diff_image))
    normalized_score = np.clip(anomaly_score / max_anomaly_score, 0, 1)

    if anomaly_score <= thresh_anomaly_score:
        label = "Pouco Desgastada"
    elif anomaly_score < 0.9:
        label = "Desgastada"
    else:
        label = "Muito Desgastada"

    fig, ax = plt.subplots(figsize=(4, 1.2))
    bar_img = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(bar_img, cmap=cmap, aspect='auto')
    ax.axvline(x=int(normalized_score * 256), color='black', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Nível de Desgaste")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    gradiente_base64 = base64.b64encode(buf.read()).decode('utf-8')

    def encode_image(img):
        _, img_encoded = cv2.imencode('.png', img)
        return base64.b64encode(img_encoded).decode('utf-8')

    os.remove(temp_input)
    os.remove(temp_output)

    return jsonify({
        'label': label,
        'score': anomaly_score,
        'original_base64': encode_image(original_image),
        'preprocessed_base64': encode_image(preprocessed_image),
        'gradiente_base64': gradiente_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)