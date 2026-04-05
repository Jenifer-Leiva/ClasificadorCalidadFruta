
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

from libs import os,cv2, np, plt, mh
import random
#--------------------------------------------------------
# EXTRACCION DE CARACTERISTICAS DE COLOR Y TEXTURA
#--------------------------------------------------------
# Características de color: Momentos de color en espacio HSV
#--------------------------------------------------------

def extraer_caracteristicas(base_dir="./DatasetFrutasSegmentadas"):
    feature_vectors = []

    for fruta in os.listdir(base_dir):
        fruta_path = os.path.join(base_dir, fruta)
        if not os.path.isdir(fruta_path):
            continue

        for estado in os.listdir(fruta_path):
            estado_path = os.path.join(fruta_path, estado)
            if not os.path.isdir(estado_path):
                continue

            files = [f for f in os.listdir(estado_path) if f.lower().endswith(".png")]

            for f in files:
                src = os.path.join(estado_path, f)
                img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    features_vector_img = []
                    features_vector_img.extend(caracteristicas_color(img))
                    features_vector_img.extend(caracteristicas_textura_GLCM(img))
                    features_vector_img.extend(caracteristicas_textura_Haralick(img))

                    #feature_vector_img = np.concatenate([[fruta],[estado], features_vector_img])
                    feature_vectors.append(features_vector_img)

    features_matrix = np.array(feature_vectors)
    print("Extracción de características de color y texture completada") 


def caracteristicas_color(img):

    # Extraer características de color
    # 1. Transformar de BRG (Por defecto en OpenCV) al espacio de color HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = img_hsv[:, :, 0]
    #S = img_hsv[:, :, 1]
    #V = img_hsv[:, :, 2]

    # h_mean = np.mean(H)
    # s_mean = np.mean(S)
    # v_mean = np.mean(V)

    # h_std = np.std(H)
    # s_std = np.std(S)
    # v_std = np.std(V)

    #Histograma para el canal de Hue
    # En OpenCV Hue está en el rango de 0 a 179
    #hist_h = cv2.calcHist([H], [0], None, [180], [0, 180])
    #hist_h_norm = hist_h / hist_h.sum()
    #hist_h_norm_flat = hist_h_norm.flatten()

    #Tonos dominantes usando KMeans
    h = H.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(h)
    dominant_hues = np.sort(kmeans.cluster_centers_.flatten())

    return dominant_hues[0], dominant_hues[1]#, hist_h_norm_flat
#--------------------------------------------------------
# Características de textura: GLCM 
#--------------------------------------------------------

def caracteristicas_textura_GLCM(img):

#--------------------------------------------------------
    #Características de textura GLCM
#--------------------------------------------------------
    #Pasar a blanco y negro
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray_image, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')
    entropy = graycoprops(glcm, 'entropy')

    # Media de los 4 ángulos
    mean_contrast = np.mean(contrast)
    mean_entropy = np.mean(entropy)

    return mean_contrast, mean_entropy

#-------------------------------------------------------
    #Características de textura Haralick
#--------------------------------------------------------

def caracteristicas_textura_Haralick(img):

    #Pasar a blanco y negro
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # labels = ["dominant_hue_1", "dominant_hue_2", "mean_contrast_GLCM", "mean_entropy_GLCM",
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
    
def prueba_caracteristicas(base_in="./DatasetFrutasSegmentadas"):

    feature_vectors = [] 
    labels = []

    for fruta in ["Banana", "Mango"]:
        for estado in ["Fresh", "Rotten"]:
            estado_path = os.path.join(base_in, fruta, estado)
            if not os.path.isdir(estado_path):
                continue

            files = [f for f in os.listdir(estado_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

            random_img = random.choice(files)

            src = os.path.join(estado_path, random_img)
            img = cv2.imread(src)
            if img is not None:
                features_vector_img = []
                features_vector_img.extend(caracteristicas_color(img))
                features_vector_img.extend(caracteristicas_textura_GLCM(img))
                features_vector_img.extend(caracteristicas_textura_Haralick(img))

                #feature_vector_img = np.concatenate([[fruta],[estado], features_vector_img])
                feature_vectors.append(features_vector_img)
                labels.append(f"{fruta}_{estado}")
                print(src)

    features_matrix = np.array(feature_vectors)
    labels = np.array(labels)

    print("Extracción de características de color y texture completada") 
    print("Matriz de características:")
    print(features_matrix)
    print(labels)