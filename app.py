import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler

sourceBasePath = os.path.dirname(os.path.abspath(__file__)) + '\\pathUnico\\'
sourceFolderPath = os.path.join(sourceBasePath, 'Treino')
destFolderPath = os.path.join(sourceBasePath, 'Teste')
save_path = destFolderPath
os.makedirs(destFolderPath, exist_ok=True)

####TROCAR PARA O NOME DA MOEDA UPADA
image_path = os.path.join(destFolderPath, 'imagem upada (exemplo).jpg')
####TROCAR PARA O ESCOLHIDO NO MENU
moeda = "10"
####DEBUG
debug = True

train_path = os.path.join(sourceBasePath, 'Treino', moeda)

def preProcess(image_path, output_path, debug=False):
    imgName = os.path.basename(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Thresholding adaptativo para segmentação robusta
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Nenhum contorno detectado em {imgName}, imagem ignorada.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(largest_contour)
    x, y, r = int(x), int(y), int(r)

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)

    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    if debug:
        debug_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_path, imgName + '_debug.png'), debug_img)

    cv2.imwrite(os.path.join(output_path, imgName), result)

preProcess(image_path, destFolderPath, debug)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img.flatten())
    return np.array(images)

train_images = load_images_from_folder(train_path)
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)

mean_normal_image = np.mean(train_images_scaled, axis=0).reshape(256, 256)

thresh_anomaly_score = 0.60  
max_anomaly_score = 1.8  

cmap = LinearSegmentedColormap.from_list("anomaly_cmap", ["green", "yellow", "red"])

image_name = os.path.basename(image_path)

if image_path is not None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    test_image_resized = cv2.resize(image, (256, 256))
    test_image_flat = scaler.transform(test_image_resized.flatten().reshape(1, -1))

    diff_image = np.abs(test_image_flat.reshape(256, 256) - mean_normal_image)
    anomaly_score = np.mean(diff_image)
    normalized_score = np.clip(anomaly_score / max_anomaly_score, 0, 1)

    if anomaly_score <= thresh_anomaly_score:
        label = "Pouco Desgastada"
    elif anomaly_score < 0.9:
        label = "Desgastada"
    else:
        label = "Muito Desgastada"

    fig = plt.figure(figsize=(5, 6))
    ax_img = fig.add_axes([0.1, 0.25, 0.8, 0.7])
    ax_img.imshow(test_image_resized, cmap='gray')
    ax_img.set_title(f"{label}\nScore: {anomaly_score:.3f}")
    ax_img.axis('off')



    # Barra de gradiente personalizada
    ax_bar = fig.add_axes([0.1, 0.1, 0.8, 0.05])
    bar_img = np.linspace(0, 1, 256).reshape(1, -1)
    ax_bar.imshow(bar_img, cmap=cmap, aspect='auto')
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_title("Nível de Anomalia", fontsize=10)

    # Indicador da posição na barra
    marker_pos = int(normalized_score * 256)
    ax_bar.plot([marker_pos, marker_pos], [0, 1], color='black', linewidth=2, transform=ax_bar.get_xaxis_transform(), clip_on=False)

    result_path = os.path.join(save_path, f"{image_name}_resultado.png")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

    print(f"Imagem {image_name}: {label} - Anomaly score: {anomaly_score:.3f}")
