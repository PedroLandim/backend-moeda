from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
from flask_cors import CORS

# Inicializa o app
app = Flask(__name__)
CORS(app)

# Config de diretórios
sourceBasePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pathUnico')
train_base_path = os.path.join(sourceBasePath, 'Treino')

# Preprocessamento (reutilizável para treino e teste)
def preProcess_image(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    (x, y), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, thickness=-1)
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    return result, mask

# Carregar e preprocessar imagens de treino

def load_and_preprocess_train_images(base_folder, coin_type):
    images = []
    folder = os.path.join(base_folder, str(coin_type))
    if not os.path.exists(folder):
        return np.array(images)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            preprocessed, _ = preProcess_image(image)
            if preprocessed is not None:
                preprocessed = cv2.resize(preprocessed, (256, 256))
                images.append(preprocessed.flatten())
    return np.array(images)

# Thresholds
thresh_anomaly_score = 1.9
max_anomaly_score = 4
cmap = LinearSegmentedColormap.from_list("anomaly_cmap", ["green", "yellow", "red"])

def preProcess(img_array):
    image = cv2.imdecode(np.frombuffer(img_array, np.uint8), cv2.IMREAD_GRAYSCALE)
    result, mask = preProcess_image(image)
    return image, result, mask

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    coin_type = request.json.get('coin_type', '50')
    img_data = base64.b64decode(data)

    train_images = load_and_preprocess_train_images(train_base_path, coin_type)
    if train_images.size == 0:
        return jsonify({'error': 'Sem imagens de treino para a moeda especificada.'}), 400

    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images)
    mean_normal_image_scaled = np.mean(train_images_scaled, axis=0).reshape(256, 256)
    mean_normal_image_raw = np.mean(train_images, axis=0).reshape(256, 256)

    original, processed, mask = preProcess(img_data)
    if processed is None:
        return jsonify({'error': 'Falha no processamento da imagem.'}), 400

    resized = cv2.resize(processed, (256, 256))
    mask_resized = cv2.resize(mask, (256, 256)) // 255
    test_flat = resized.flatten().reshape(1, -1)
    test_flat_scaled = scaler.transform(test_flat)
    diff_image_scaled = np.abs(test_flat_scaled.reshape(256, 256) - mean_normal_image_scaled)
    diff_image_raw = np.abs(resized.astype(np.float32) - mean_normal_image_raw)
    diff_image_raw *= mask_resized

    anomaly_score = float(np.mean(diff_image_scaled))
    normalized_score = np.clip(anomaly_score / max_anomaly_score, 0, 1)

    if anomaly_score <= thresh_anomaly_score:
        label = "Pouco Desgastada"
    elif anomaly_score < 2.9:
        label = "Desgastada"
    else:
        label = "Muito Desgastada"

    # Gerar imagem da barra
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

    # Gerar mapa de diferença absoluta (diff map)
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    diffmap = ax3.imshow(diff_image_raw, cmap='hot', interpolation='nearest')
    plt.colorbar(diffmap, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("Mapa de Diferença Absoluta")
    ax3.axis('off')

    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight')
    plt.close(fig3)
    buf3.seek(0)
    diffmap_base64 = base64.b64encode(buf3.read()).decode('utf-8')

    # Gerar heatmap da diferença visual real
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    heatmap = ax2.imshow(diff_image_raw, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("Mapa de Calor")
    ax2.axis('off')

    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    heatmap_base64 = base64.b64encode(buf2.read()).decode('utf-8')

    def encode_image(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'label': label,
        'score': anomaly_score,
        'original_base64': encode_image(original),
        'preprocessed_base64': encode_image(processed),
        'gradiente_base64': gradiente_base64,
        'heatmap_base64': heatmap_base64,
        'diffmap_base64': diffmap_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
