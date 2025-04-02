import os
import cv2
import numpy as np

sourceBasePath = os.path.dirname(os.path.abspath(__file__)) + '\\Cruzeiro_Novo\\'
subfolders = ['1', '2', '5', '10', '20', '50', 'verso']

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

for folder in ['treino', 'teste']:
    for subfolder in subfolders:
        sourceFolderPath = os.path.join(sourceBasePath, folder, subfolder)
        destFolderPath = os.path.join(sourceBasePath, f"output {folder}", subfolder)

        os.makedirs(destFolderPath, exist_ok=True)
        
        for item in os.listdir(sourceFolderPath):
            image_path = os.path.join(sourceFolderPath, item)
            preProcess(image_path, destFolderPath, debug=True)
