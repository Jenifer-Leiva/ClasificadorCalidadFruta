from cv2.gapi import mask
import os, cv2, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analizar_forma(img_path):
    """
    Calcula métricas de forma a partir de la máscara de una fruta en una imagen.
    Devuelve un diccionario con área, perímetro, circularidad, compacidad y excentricidad.
    """

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Si tiene canal alfa, usarlo como máscara
    if img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)

    # Buscar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "archivo": os.path.basename(img_path),
            "area": None,
            "perimetro": None,
            "circularidad": None,
            "compacidad": None,
            "excentricidad": None
        }

    # Tomar el contorno más grande
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Métricas geométricas
    circularidad = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    compacidad = area / perimeter if perimeter > 0 else 0

    # Excentricidad (ajuste de elipse)
    excentricidad = None
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (center, axes, angle) = ellipse
        a = max(axes) / 2.0
        b = min(axes) / 2.0
        excentricidad = np.sqrt(1 - (b**2 / a**2))

    # Momentos de Hu
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Escala logarítmica (estándar)
    hu_moments = [-np.sign(h) * np.log10(abs(h)) if h != 0 else 0 for h in hu_moments]


    return {
        "archivo": os.path.basename(img_path),
        "area": area,
        "perimetro": perimeter,
        "circularidad": circularidad,
        "compacidad": compacidad,
        "excentricidad": excentricidad,

        # 🔥 nuevos
        "hu1": hu_moments[0],
        "hu2": hu_moments[1],
        "hu3": hu_moments[2],
        "hu4": hu_moments[3],
        "hu5": hu_moments[4],
        "hu6": hu_moments[5],
        "hu7": hu_moments[6],
    }

def extraer_caracteristicas_forma(base_dir="./DatasetFrutasAumentadas", out_dir="./galeria_resultados"):
    out_forma_dir = os.path.join(out_dir, "forma")
    os.makedirs(out_forma_dir, exist_ok=True)

    resultados = []
    for subset in ["train", "val"]:
        subset_path = os.path.join(base_dir, subset)
        if not os.path.isdir(subset_path):
            continue

        for class_name in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for f in os.listdir(class_path):
                if f.endswith(".png"):
                    img_path = os.path.join(class_path, f)
                    metrics = analizar_forma(img_path)
                    if metrics:
                        metrics["subset"] = subset
                        metrics["clase"] = class_name
                        resultados.append(metrics)

    # Convertir a DataFrame
    df = pd.DataFrame(resultados)

    # Lista de características a graficar
    features = [
    "area", "perimetro", "circularidad", "compacidad", "excentricidad",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"
    ]

    for feat in features:
        plt.figure(figsize=(8,6))
        sns.boxplot(x="clase", y=feat, data=df, hue="subset")
        sns.stripplot(x="clase", y=feat, data=df, hue="subset", 
                      dodge=True, alpha=0.3, color="black")  # puntos individuales
        plt.title(f"Distribución de {feat} por clase")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.ylim(-10, 10)

        # Guardar gráfico
        out_file = os.path.join(out_forma_dir, f"{feat}_boxplot.png")
        plt.savefig(out_file)
        plt.close()

    print(f"✅ Boxplots de características de forma guardados en {out_forma_dir}")


