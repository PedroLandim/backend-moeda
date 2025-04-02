import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler

moeda = "10"  # ajuste conforme necessário
base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "Cruzeiro_Novo", "output treino", moeda)
test_image_path = os.path.join(base_path, "Cruzeiro_Novo", "output teste", moeda)
save_path = os.path.join(base_path, "Cruzeiro_Novo", "output resultados", moeda)

if not os.path.exists(save_path):
    os.makedirs(save_path)

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

for image_name in os.listdir(test_image_path):
    image_path = os.path.join(test_image_path, image_name)
    test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if test_image is not None:
        test_image_resized = cv2.resize(test_image, (256, 256))
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