
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

from libs import os,cv2, np, plt, mh, sns,pd
from sklearn.preprocessing import LabelEncoder
import random
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

        return rgb, mask
    
    #return np.dstack((rgb_aug, alpha))

def caracteristicas_color(img, mask):

    # Extraer características de color
    # 1. Transformar al espacio de color HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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

def caracteristicas_textura_GLCM(img, mask):

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

def caracteristicas_textura_Haralick(img, mask):

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
    
def diagramas(base_in="./DatasetFrutasAumentadas", out_dir="./galeria_resultados", features_matrix=None):

    #Visualización de características

    for column in features_matrix.columns:  # Saltar las dos primeras columnas (fruit y state)
        if column == "fruit" or column == "state":
            continue
    
    column_index = features_matrix.columns.get_loc(column)

    banano = features_matrix[features_matrix[:, 0] == 0]
    mango = features_matrix[features_matrix[:, 0] == 1]
    fresco = features_matrix[features_matrix[:, 0] == 0]
    podrido = features_matrix[features_matrix[:, 0] == 1]

    d_1 = banano [:, column_index]
    d_2 = mango[:, column_index]
    d_3 = fresco[:, column_index]
    d_4 = podrido[:, column_index]

    d12 = [d_1, d_2]
    d34 = [d_3, d_4]

    #Tipo de fruta
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])

    bp = ax.boxplot(d12, patch_artist=True)
    colors = ['lightyellow', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(['Banano', 'Mango'])
    plt.title(f'Box Plot de {column} por tipo de fruta')
    plt.ylabel(column)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_TipoFruta.png"))
    #plt.show()

    plt.figure(figsize=(10, 7))
    sns.violinplot(data=d12, palette=colors)
    plt.title(f'Violin Plot de {column} por Tipo de fruta')
    plt.ylabel(column)
    plt.xticks(ticks=[0, 1], labels=['Banano', 'Mango'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_TipoFruta.png"))
    #plt.show()

    #Estado de la fruta
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])

    bp = ax.boxplot(d34, patch_artist=True)
    colors = ['lightgreen', 'lightred']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(['Fresco', 'Podrido'])
    plt.title(f'Box Plot de {column} por Estado de la fruta')
    plt.ylabel(column)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_EstadoFruta.png"))
    #plt.show()

    plt.figure(figsize=(10, 7))
    sns.violinplot(data=d34, palette=colors)
    plt.title(f'Violin Plot de {column} por Estado de la fruta')
    plt.ylabel(column)
    plt.xticks(ticks=[0, 1], labels=['Fresco', 'Podrido'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_EstadoFruta.png"))
    #plt.show()


def prueba_caracteristicas(base_in="./DatasetFrutasAumentadas", out_dir="./galeria_resultados"):

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
                rgb, mask = aplicar_en_mascara(img)
                features_vector_img = []
                features_vector_img.append(fruta)
                features_vector_img.append(estado)
                features_vector_img.extend(caracteristicas_color(rgb, mask))
                features_vector_img.extend(caracteristicas_textura_GLCM(rgb, mask))
                features_vector_img.extend(caracteristicas_textura_Haralick(rgb, mask))

                #feature_vector_img = np.concatenate([[fruta],[estado], features_vector_img])
                feature_vectors.append(features_vector_img)
                #labels.append(f"{fruta}_{estado}")
                print(src)

    #labels = np.array(labels)
    #features_matrix = np.array(feature_vectors)
    features_matrix = pd.DataFrame(feature_vectors, columns=["fruit","state","dominant_hue_1", "dominant_hue_2", 
                                                                "mean_contrast_GLCM", "mean_entropy_GLCM",
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
                                                                "Information Measure of Correlation 2"  ])  
    
    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    le =LabelEncoder()
    features_matrix["fruit"] = le.fit_transform(features_matrix["fruit"])
    features_matrix["state"] = le.fit_transform(features_matrix["state"])

    print(features_matrix.head())
    features_matrix.to_csv(os.path.join(out_dir, "MatrizCaracteristicas.csv"), index=False)
    
    diagramas(base_in, "./galeria_resultados/Diagramas_pruebas", features_matrix)

    print("Extracción de características de color y textura de prueba completada") 
    print("Matriz de características:")

    #print(features_matrix)
    #print(labels)

def feature_selection(base_in="./DatasetFrutasAumentadas", out_dir="./galeria_resultados"):
    pass

def extraer_caracteristicas(base_dir="./DatasetFrutasAumentadas", out_dir="./galeria_resultados"):
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
                    rgb, mask = aplicar_en_mascara(img)
                    features_vector_img = []
                    features_vector_img.append(fruta)
                    features_vector_img.append(estado)
                    features_vector_img.extend(caracteristicas_color(rgb, mask))
                    features_vector_img.extend(caracteristicas_textura_GLCM(rgb, mask))
                    features_vector_img.extend(caracteristicas_textura_Haralick(rgb, mask))

                    #feature_vector_img = np.concatenate([[fruta],[estado], features_vector_img])
                    feature_vectors.append(features_vector_img)

    #features_matrix = np.array(feature_vectors)
    features_matrix = pd.DataFrame(feature_vectors, columns=["fruit","state","dominant_hue_1", "dominant_hue_2", 
                                                                "mean_contrast_GLCM", "mean_entropy_GLCM",
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
                                                                "Information Measure of Correlation 2"  ])  
    
    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    le =LabelEncoder()
    features_matrix["fruit"] = le.fit_transform(features_matrix["fruit"])
    features_matrix["state"] = le.fit_transform(features_matrix["state"])

    print(features_matrix.head())
    features_matrix.to_csv(os.path.join(out_dir, "MatrizCaracteristicas.csv"), index=False)
    diagramas(base_dir, out_dir, features_matrix)

    print("Extracción de características de color y texture completada") 

