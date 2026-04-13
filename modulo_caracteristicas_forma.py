import os, cv2, numpy as np

def analizar_forma(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Si tiene canal alfa, usarlo como máscara
    if img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        # convertir a gris y binarizar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)

    # Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Circularidad
    circularidad = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

    # Compacidad (área/perímetro)
    compacidad = area / perimeter if perimeter > 0 else 0

    # Excentricidad (ajuste de elipse)
    excentricidad = None
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (center, axes, angle) = ellipse
        a = max(axes) / 2.0
        b = min(axes) / 2.0
        excentricidad = np.sqrt(1 - (b**2 / a**2))

    return {
        "area": area,
        "perimetro": perimeter,
        "circularidad": circularidad,
        "compacidad": compacidad,
        "excentricidad": excentricidad
    }

# 
base_dir = "./DatasetFrutasAumentadas/train/banana_fresh"
for f in os.listdir(base_dir):
    if f.endswith(".png"):
        metrics = analizar_forma