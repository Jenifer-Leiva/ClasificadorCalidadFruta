
from libs import os, cv2, np, plt
from sklearn.cluster import KMeans
#--------------------------------------------------------
# PREPROCESAMIENTO
# Limpieza de ruido
#--------------------------------------------------------

def limpiar_ruido(img_flat, size=128):
    img = img_flat.reshape(size, size, 3)
    img_gauss = cv2.GaussianBlur(img, (3, 3), 0)
    img_gauss_median = cv2.medianBlur(img_gauss, 5)
    return img_gauss_median

#--------------------------------------------------------
# FUNCIONES SEGEMENTACION K MEANS
#--------------------------------------------------------
# Seleccion de cluster por media (fondo claro)
def auto_cluster_kmean(kmeans_model):
    cluster_centers = kmeans_model.cluster_centers_
    mean_values = cluster_centers.mean(axis=1)
    background_cluster_index = np.argmax(mean_values) 
    return background_cluster_index

# Eliminar componentes pequeños
def filtrar_componentes(mask, min_area=500):
    # Encontrar componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Crear máscara nueva
    mask_filtrada = np.zeros_like(mask)
    for i in range(1, num_labels):  # 0 es fondo
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask_filtrada[labels == i] = 255
    return mask_filtrada

#--------------------------------------------------------
# FUNCIONES RECORTAR
#--------------------------------------------------------
def resize_with_padding(img, size=128):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Crear lienzo cuadrado
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    # Centrar
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def recortar_fruta(img_rgb, mask, size=128, margin=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img_rgb, (size, size))  # fallback

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Expandir con margen
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_rgb.shape[1] - x, w + 2*margin)
    h = min(img_rgb.shape[0] - y, h + 2*margin)

    # Forzar proporción cuadrada
    side = max(w, h)
    x = max(0, x - (side - w)//2)
    y = max(0, y - (side - h)//2)
    w = side
    h = side

    cropped = img_rgb[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_RGB2RGBA)
    rgba[..., 3] = mask_crop

    return resize_with_padding(rgba, size=size)


#--------------------------------------------------------
# SEGEMENTACION K MEANS
#--------------------------------------------------------

# Función principal de segmentación 
# --- segmentar_guardar ---
# --- segmentar_guardar ---
def segmentar_guardar(img_flat, size=128, output_dir="./DatasetFrutasSegmentadas", filename="fruta_segmentada.png"):
    img_rgb = limpiar_ruido(img_flat, size=size).astype(np.uint8)

    # KMeans clustering
    img_data = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(img_data)
    segmented_labels = kmeans.labels_
    background_cluster_index = auto_cluster_kmean(kmeans)

    fruit_mask = (segmented_labels != background_cluster_index).reshape(size, size).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    fruit_mask_filled = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
    fruit_mask_filled = filtrar_componentes(fruit_mask_filled, min_area=500)

    # Overlay solo para visualización
    mask_color = np.zeros_like(img_rgb)
    mask_color[fruit_mask_filled == 255] = [255, 0, 0]
    overlay = cv2.addWeighted(img_rgb, 1.0, mask_color, 0.5, 0)

    # Recorte con padding
    fruta_recortada = recortar_fruta(img_rgb, fruit_mask_filled, size=size)

    os.makedirs(output_dir, exist_ok=True)
    recorte_path = os.path.join(output_dir, filename.replace("_seg.png", "_crop.png"))
    cv2.imwrite(recorte_path, cv2.cvtColor(fruta_recortada, cv2.COLOR_RGBA2BGRA))

    return img_rgb, overlay, fruta_recortada



#--------------------------------------------------------
# CARPETA DATASET FRUTAS SEGMENTADAS
#--------------------------------------------------------

# --- division_segmentacion ---
def division_segmentacion(base_in="./DatasetFrutas", base_out="./DatasetFrutasSegmentadas", size=128):
    ejemplos = []  # lista para guardar un ejemplo por clase

    for fruta in ["Banana", "Mango"]:
        for estado in ["Fresh", "Rotten"]:
            estado_path = os.path.join(base_in, fruta, estado)
            if not os.path.isdir(estado_path):
                continue

            files = [f for f in os.listdir(estado_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

            ejemplo_mostrado = False
            for i, f in enumerate(files):
                img_path = os.path.join(estado_path, f)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (size, size))
                img_flat = img.flatten()

                out_dir = os.path.join(base_out, fruta, estado)
                os.makedirs(out_dir, exist_ok=True)
                filename = f"{fruta}_{estado}_{i}_seg.png"

                original, overlay, recorte = segmentar_guardar(img_flat, size=size, output_dir=out_dir, filename=filename)

                # Guardar un ejemplo por clase
                if not ejemplo_mostrado:
                    ejemplos.append((fruta, estado, original, overlay, recorte))
                    ejemplo_mostrado = True

    # Mostrar todos los ejemplos en una sola ventana fija
    n = len(ejemplos)
    plt.figure(figsize=(12, 4*n))
    for idx, (fruta, estado, orig, seg, rec) in enumerate(ejemplos):
        plt.subplot(n, 3, idx*3+1); plt.imshow(orig); plt.title(f"{fruta}-{estado} Original"); plt.axis("off")
        plt.subplot(n, 3, idx*3+2); plt.imshow(seg); plt.title("Segmentada"); plt.axis("off")
        plt.subplot(n, 3, idx*3+3); plt.imshow(rec); plt.title("Recortada"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("Segmentación aplicada a todas las imágenes por clase")

