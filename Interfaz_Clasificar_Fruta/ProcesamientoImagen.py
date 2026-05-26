import json
from PIL import features
from cv2 import kmeans
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.feature_extraction import image

from core.libs import os,cv2, np, plt, mh, sns,pd, joblib# Para correción
from sklearn.preprocessing import StandardScaler


#--------------------------------------------------------
# EXTRACCION DE CARACTERISTICAS DE COLOR Y TEXTURA
#--------------------------------------------------------
# Características de color: Momentos de color en espacio HSV
#--------------------------------------------------------

def aplicar_en_mascara(img):
    if img.shape[2] == 4:  # RGBA
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]

        mask = alpha > 0

        rgb_orig = rgb.copy()
        #rgb_mod = funcion_feature(rgb)

        # Aplicar solo donde hay fruta
        #rgb_aug[mask] = rgb_mod[mask]

        return rgb, mask, alpha

    # Soporte para imágenes RGB/BGR sin canal alfa
    rgb = img[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    alpha = np.full(gray.shape, 255, dtype=np.uint8)
    return rgb, mask.astype(bool), alpha

def caracteristicas_color_hsv(img, mask, alpha):

    #Suavizar para hacer más facil la segmentación 
    img_gauss = cv2.GaussianBlur(img, (3, 3), 0)
    img_gauss_median = cv2.medianBlur(img_gauss, 5)

    # Extraer características de color
    # 1. Transformar al espacio de color HSV
    img_hsv = cv2.cvtColor(img_gauss_median, cv2.COLOR_RGB2HSV)
    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]

    #Evitar valores de flash o fondo
    #mask_complete = mask & (S > 20) & (V > 20) & (V < 240)
    mask_complete = mask

    img_hsv_masked = img_hsv[mask_complete]

    #Tonos dominantes usando KMeans
    hsv_kmeans = img_hsv_masked.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(hsv_kmeans)

    labels = kmeans.labels_
    counts = np.bincount(labels)
    ordered_by_importance = np.argsort(counts)[::-1] #De mayor a menor
    
    two_main_clusters = ordered_by_importance[:2]

    #Solo se tienen en cuentan los tonos de los 2 clusters más importantes (más cantidad de pixeles), dado que el de menor probablemente son partes que quedaron del fondo o flash.
    centers = kmeans.cluster_centers_
    dominant_hues = centers[two_main_clusters, 0]  # Ordenar por hue
    Saturations = centers[two_main_clusters, 1]
    Values = centers[two_main_clusters, 2]
    
    # Reconstruir imagen de clusters
    segmented = np.zeros_like(img_hsv, dtype=np.float32)
    segmented[mask_complete] = centers[labels]
    #segmented[~mask_complete] = kmeans.cluster_centers_[0]

    segmented_hsv = segmented.astype(np.uint8)
    segmented_rgb = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2RGB)
    segmented_rgba = np.dstack((segmented_rgb, alpha))
    segmented_rgba_display = cv2.cvtColor(segmented_rgba, cv2.COLOR_RGBA2BGRA)

    img_alpha = np.dstack((img, alpha))
    img_display = cv2.cvtColor(img_alpha, cv2.COLOR_RGBA2BGRA)

    # Visualización
    """plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title("Imagen original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgba_display)
    plt.title("Clusters de color (K-Means)")
    plt.axis('off')

    plt.tight_layout()
    #plt.savefig(os.path.join("./galeria_resultados/Diagramas_pruebas", f"segmentacion_{fruta_estado}.png"))  

    plt.show()"""

    # #print(np.unique(segmented_values_H))
    # print("Tonos dominantes (Hue):", dominant_hues)
    # print("Tonos dominantes (HSV):", centers[two_main_clusters])

    return dominant_hues[0], dominant_hues[1], Saturations[0], Saturations[1], Values[0], Values[1]#, hist_h_norm_flat

#--------------------------------------------------------
# Características de textura: GLCM 
#--------------------------------------------------------

def caracteristicas_textura_GLCM(img, mask):
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image[~mask] = 0

    glcm = graycomatrix(gray_image, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')
    entropy = graycoprops(glcm, 'entropy')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    correlation = graycoprops(glcm, 'correlation')

    # Media de los 4 ángulos
    mean_contrast = np.mean(contrast)
    mean_entropy = np.mean(entropy)
    mean_energy = np.mean(energy)
    mean_homogeneity = np.mean(homogeneity)
    mean_correlation = np.mean(correlation)

    return mean_contrast, mean_entropy, mean_energy, mean_homogeneity, mean_correlation

#------------------------------------------------------
    #Características de textura Haralick
#--------------------------------------------------------

def caracteristicas_textura_Haralick(img, mask):

    #Pasar a blanco y negro
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image[~mask] = 0
    # labels = ["dominant_hue_1", "dominant_hue_2", "mean_contrast_GLCM", "mean_entropy_GLCM", "correlation_GLCM", "energy_GLCM", "homogeneity_GLCM",
    # "Angular Second Moment",
    # "Contrast",
    # "Correlation",
    # "Sum of Squares (Variance)",
    # "Inverse Difference Moment",
    # "Sum Average",
    # "Sum Variance",
    # "Sum Entropy",
    # "Entropy",
    # "Difference Variance",
    # "Difference Entropy",
    # "Information Measure of Correlation 1",
    # "Information Measure of Correlation 2"
    # ]

    haralick_features =  mh.features.haralick(gray_image, distance=5, return_mean = True) 

    return haralick_features
    
# =========================================================
# EXTRACCIÓN DE CARACTERÍSTICAS DE FORMA
# =========================================================

def analizar_forma(img):

    # -----------------------------------------------------
    # MÁSCARA
    # -----------------------------------------------------

    if len(img.shape) == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0

    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)

    # -----------------------------------------------------
    # CONTORNOS
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # -----------------------------------------------------
    # MÉTRICAS GEOMÉTRICAS
    # -----------------------------------------------------

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(cnt, True)

    circularidad = (
        (4 * np.pi * area) / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    compacidad = (
        area / perimeter
        if perimeter > 0 else 0
    )

    # -----------------------------------------------------
    # EXCENTRICIDAD
    # -----------------------------------------------------

    excentricidad = 0

    if len(cnt) >= 5:

        ellipse = cv2.fitEllipse(cnt)

        (_, axes, _) = ellipse

        a = max(axes) / 2.0
        b = min(axes) / 2.0

        excentricidad = np.sqrt(1 - (b ** 2 / a ** 2))

    # -----------------------------------------------------
    # MOMENTOS DE HU
    # -----------------------------------------------------

    moments = cv2.moments(mask)

    hu_moments = cv2.HuMoments(moments).flatten()

    # Escala logarítmica
    hu_moments = [
        -np.sign(h) * np.log10(abs(h))
        if h != 0 else 0
        for h in hu_moments
    ]

    return area, perimeter, circularidad, compacidad, excentricidad, hu_moments[0],hu_moments[1],hu_moments[2],hu_moments[3], hu_moments[4], hu_moments[5], hu_moments[6]


def ProcesarImagen(img):

    #Segmentar
  

    src = img

    # =====================================================
    # CARGAR IMAGEN
    # =====================================================

    imag = cv2.imread(src, cv2.IMREAD_COLOR)

    if imag is None:
        return None

    imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

    imag = cv2.resize(imag, (128, 128))

    # =====================================================
    # LIMPIEZA DE RUIDO
    # =====================================================

    img_gauss = cv2.GaussianBlur(imag, (3, 3), 0)
    img_gauss_median = cv2.medianBlur(img_gauss, 5)

    # =====================================================
    # KMEANS SEGMENTACION
    # =====================================================

    img_data = img_gauss_median.reshape(-1, 3)

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    kmeans.fit(img_data)

    segmented_labels = kmeans.labels_

    # Detectar fondo automáticamente
    cluster_centers = kmeans.cluster_centers_
    mean_values = cluster_centers.mean(axis=1)

    background_cluster_index = np.argmax(mean_values)

    # Máscara fruta
    fruit_mask = (
        segmented_labels != background_cluster_index
    ).reshape(128, 128).astype(np.uint8) * 255

    # =====================================================
    # LIMPIEZA MORFOLOGICA
    # =====================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    fruit_mask = cv2.morphologyEx(
        fruit_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =====================================================
    # RELLENAR HUECOS
    # =====================================================

    im_floodfill = fruit_mask.copy()

    h, w = fruit_mask.shape[:2]

    mask_flood = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(
        im_floodfill,
        mask_flood,
        (0, 0),
        255
    )

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    fruit_mask = fruit_mask | im_floodfill_inv

    # =====================================================
    # ELIMINAR COMPONENTES PEQUEÑOS
    # =====================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fruit_mask,
        connectivity=8
    )

    mask_filtrada = np.zeros_like(fruit_mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 500:
            mask_filtrada[labels == i] = 255

    fruit_mask = mask_filtrada

    # =====================================================
    # CREAR IMAGEN RGBA
    # =====================================================

    rgba = cv2.cvtColor(imag, cv2.COLOR_RGB2RGBA)

    rgba[..., 3] = fruit_mask

    
    src = img
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    # =====================================================
    # EXTRAER FEATURES
    # =====================================================

    rgb, mask, alpha = aplicar_en_mascara(img)

    features_vector_img = []

    features_vector_img.extend(
        caracteristicas_color_hsv(rgb, mask, alpha)
    )

    features_vector_img.extend(
        caracteristicas_textura_GLCM(rgb, mask)
    )

    features_vector_img.extend(
        caracteristicas_textura_Haralick(rgb, mask)
    )

    features_vector_img.extend(
        analizar_forma(rgb)
    )

    

    features_matrix = pd.DataFrame([features_vector_img], columns=["dominant_hue_1", "dominant_hue_2",
                                                                "Saturation_1", "Saturation_2", "Value_1", "Value_2",
                                                                "mean_contrast_GLCM", "mean_entropy_GLCM",
                                                                "mean_energy_GLCM", "mean_homogeneity_GLCM", "mean_correlation_GLCM",
                                                                "Angular Second Moment","Contrast",
                                                                "Correlation",
                                                                "Sum of Squares (Variance)",
                                                                "Inverse Difference Moment",
                                                                "Sum Average",
                                                                "Sum Variance",
                                                                "Sum Entropy",
                                                                "Entropy",
                                                                "Difference Variance",
                                                                "Difference Entropy",
                                                                "Information Measure of Correlation 1",
                                                                "Information Measure of Correlation 2", "Area", "Perimetro",
                                                                "Circularidad","Compacidad","Excentricidad","hu1","hu2","hu3",
                                                                "hu4","hu5","hu6","hu7"]) 

    #Normalizar características 
    scaler = joblib.load("Interfaz_Clasificar_Fruta/scaler.pkl")
    numeric_cols = features_matrix.columns
    #Matrices especificas para cada tipo de fruta y estado
    features_matrix_norm = features_matrix.copy()
    features_matrix_norm[numeric_cols] = scaler.transform(features_matrix[numeric_cols])

    features_select_fruit = pd.read_csv("Interfaz_Clasificar_Fruta/features_selected_fruit.csv")
    features_selected_fruit = features_select_fruit["feature"].tolist()

    features_select_state = pd.read_csv("Interfaz_Clasificar_Fruta/features_selected_state.csv")
    features_selected_state = features_select_state["feature"].tolist()

    features_matrix_filtrada_fruit = features_matrix_norm.drop(columns=[col for col in features_matrix_norm.columns if col not in features_selected_fruit])
    
    features_matrix_filtrada_state = features_matrix_norm.drop(columns=[col for col in features_matrix_norm.columns if col not in features_selected_state])

    print(features_matrix_filtrada_fruit.head())
    print(features_matrix_filtrada_state.head())

    return features_matrix_filtrada_fruit, features_matrix_filtrada_state

def PrediccionModelo(features_matrix_filtrada_fruit, features_matrix_filtrada_state):

    # Cargar el modelo entrenado para fruta
    model_fruit = joblib.load("Interfaz_Clasificar_Fruta/trained_model_fruit.pkl")
    le_fruit= joblib.load("Interfaz_Clasificar_Fruta/label_encoder_fruit.pkl")

    prediction_fruit = model_fruit.predict(features_matrix_filtrada_fruit)
    probabilities_fruit = model_fruit.predict_proba(features_matrix_filtrada_fruit)

    prediction_fruit= le_fruit.inverse_transform(prediction_fruit)

    print(f"Predicted Class: {prediction_fruit[0]}")
    print(f"Prediction Probabilities: {probabilities_fruit[0]}")

    #Cargar el modelo entrenadeo para estado

    model_state = joblib.load("Interfaz_Clasificar_Fruta/trained_model_state.pkl")
    le_state = joblib.load("Interfaz_Clasificar_Fruta/label_encoder_state.pkl")

    prediction_state = model_state.predict(features_matrix_filtrada_state)
    probabilities_state = model_state.predict_proba(features_matrix_filtrada_state)

    prediction_state = le_state.inverse_transform(prediction_state)

    print(f"Predicted Class: {prediction_state[0]}")
    print(f"Prediction Probabilities: {probabilities_state[0]}")

    return prediction_fruit, probabilities_fruit, prediction_state, probabilities_state
