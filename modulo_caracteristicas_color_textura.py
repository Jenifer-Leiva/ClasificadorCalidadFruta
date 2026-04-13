
from cv2 import kmeans
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

from libs import os,cv2, np, plt, mh, sns,pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.preprocessing import MinMaxScaler
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
    
    #return np.dstack((rgb_aug, alpha))

def caracteristicas_color_hsv(img, mask, alpha, fruta_estado="Fruta_Estado"):

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
    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(img_display)
    # plt.title("Imagen original")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(segmented_rgba_display)
    # plt.title("Clusters de color (K-Means)")
    # plt.axis('off')

    #plt.tight_layout()
    #plt.savefig(os.path.join("./galeria_resultados", f"segmentacion_{fruta_estado}.png"))  

    #plt.show()

    #print(np.unique(segmented_values_H))
    #print("Tonos dominantes (Hue):", dominant_hues)
    #print("Tonos dominantes (HSV):", centers[two_main_clusters])

    return dominant_hues[0], dominant_hues[1], Saturations[0], Saturations[1], Values[0], Values[1]#, hist_h_norm_flat

def caracteristicas_color(img, mask, alpha):

    img_gauss = cv2.GaussianBlur(img, (3, 3), 0)
    img_gauss_median = cv2.medianBlur(img_gauss, 5)

    # Extraer características de color
    # 1. Transformar al espacio de color HSV
    img_hsv = cv2.cvtColor(img_gauss_median, cv2.COLOR_RGB2HSV)
    H = img_hsv[:, :, 0]
    #H_mask = H[mask]
    S = img_hsv[:, :, 1]
    S[~mask] = 0
    V = img_hsv[:, :, 2]
    V[~mask] = 0

    #Evitar valores de flash o fondo
    #mask_valid[~mask] = 0
    mask_complete = mask & (S > 20) & (V > 20) & (V < 240)
    #mask_complete = mask & mask_valid

    H_mask = H[mask_complete]

    #HS_mask = img_hsv[mask_complete][:, :2]

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
    h = H_mask.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(h)
    dominant_hues = np.sort(kmeans.cluster_centers_.flatten())

    labels = kmeans.labels_

    # Reconstruir imagen de clusters
    segmented_H = np.zeros_like(H, dtype=float)
    segmented_values_H = kmeans.cluster_centers_[labels].flatten()
    segmented_H[mask_complete] = segmented_values_H
    segmented_H[~mask_complete] = kmeans.cluster_centers_[0][0]

    segmented_hsv = np.dstack((segmented_H, S, V)).astype(np.uint8)
    segmented_rgb = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2RGB)
    segmented_rgba = np.dstack((segmented_rgb, alpha))
    segmented_rgba_display = cv2.cvtColor(segmented_rgba, cv2.COLOR_RGBA2BGRA)

    img_alpha = np.dstack((img, alpha))
    img_display = cv2.cvtColor(img_alpha, cv2.COLOR_RGBA2BGRA)

    # Visualización
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title("Imagen original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgba_display)
    plt.title("Clusters de color (K-Means)")
    plt.axis('off')

    plt.show()

    print(np.unique(segmented_values_H))

    return dominant_hues[0], dominant_hues[1]#, hist_h_norm_flat
#--------------------------------------------------------
# Características de textura: GLCM 
#--------------------------------------------------------

def caracteristicas_textura_GLCM(img, mask):

#--------------------------------------------------------
    #Características de textura GLCM
#--------------------------------------------------------
    #Pasar a blanco y negro
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image[~mask] = 0

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
    gray_image[~mask] = 0
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

    #banano_fresco = features_matrix[(features_matrix["fruit"] == 0) & (features_matrix["state"] == 0)]
    #banano_podrido = features_matrix[(features_matrix["fruit"] == 0) & (features_matrix["state"] == 1)]
    #mango_fresco = features_matrix[(features_matrix["fruit"] == 1) & (features_matrix["state"] == 0)]
    #mango_podrido = features_matrix[(features_matrix["fruit"] == 1) & (features_matrix["state"] == 1)]

    # banano_fresco = features_matrix[(features_matrix["fruit_state"] == 0)]
    # banano_podrido = features_matrix[(features_matrix["fruit_state"] == 1)]
    # mango_fresco = features_matrix[(features_matrix["fruit_state"] == 2)]
    # mango_podrido = features_matrix[(features_matrix["fruit_state"] == 3)]

    banano = features_matrix[features_matrix["fruit"] == 0]
    mango = features_matrix[features_matrix["fruit"] == 1]
    banano_fresco = banano[banano["state"] == 0]
    banano_podrido = banano[banano["state"] == 1]
    mango_fresco = mango[mango["state"] == 0]   
    mango_podrido = mango[mango["state"] == 1]
    #fresco = features_matrix[features_matrix["state"] == 0]
    #podrido = features_matrix[features_matrix["state"] == 1]

    for column in features_matrix.columns:  # Saltar las dos primeras columnas (fruit y state)
        if column == "fruit" or column == "state" or column == "fruit_state":
            continue
    
        #column_index = features_matrix.columns.get_loc(column)

        d_1 = banano[column]
        d_2 = mango[column]

        d_3 = banano_fresco[column]
        d_4 = banano_podrido[column]

        d_5 = mango_fresco[column]
        d_6 = mango_podrido[column]

        #d_1 = banano_fresco[column]
        #d_2 = banano_podrido[column]
        #d_3 = mango_fresco[column]
        #d_4 = mango_podrido[column]


        d12 = [d_1, d_2] # Banano vs Mango
        d34 = [d_3, d_4] # Banano fresco vs podrido
        d56 = [d_5, d_6] # Mango fresco vs podrido

        #d = [d_1, d_2, d_3, d_4] # Banano fresco, Banano podrido, Mango fresco, Mango podrido

        #Tipo de fruta
        # fig = plt.figure(figsize =(10, 7))
        # ax = fig.add_axes([0, 0, 1, 1])

        # bp = ax.boxplot(d, patch_artist=True)
        # colors = ['lightyellow', 'lightgreen']
        # for patch, color in zip(bp['boxes'], colors):
        #     patch.set_facecolor(color)

        # ax.set_xticklabels(['Banano_fresco', 'Banano_podrido', 'Mango_fresco', 'Mango_podrido'])
        # plt.title(f'Box Plot de {column} por tipo y estado de fruta')
        # plt.ylabel(column)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_TipoEstadoFruta.png"), bbox_inches='tight')
        # #plt.show()

        # plt.figure(figsize=(10, 7))
        # sns.violinplot(data=d)
        # plt.title(f'Violin Plot de {column} por tipo y estado de fruta')
        # plt.ylabel(column)
        # plt.xticks(ticks=[0, 1, 2, 3], labels=['Banano_fresco', 'Banano_podrido', 'Mango_fresco', 'Mango_podrido'])
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_TipoEstadoFruta.png"))
        # #plt.show()

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
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_TipoFruta.png"), bbox_inches='tight')
        #plt.show()

        # plt.figure(figsize=(10, 7))
        # sns.violinplot(data=d12)
        # plt.title(f'Violin Plot de {column} por tipo y estado de fruta')
        # plt.ylabel(column)
        # plt.xticks(ticks=[0, 1], labels=['Banano', 'Mango'])
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_TipoEstadoFruta.png"))
        # #plt.show()

        # #Estado de la fruta: Banano fresco vs podrido
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])

        bp = ax.boxplot(d34, patch_artist=True)
        colors = ['lightyellow', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticklabels(['Fresco', 'Podrido'])
        plt.title(f'Box Plot de {column} por Estado de la fruta (Banano)')
        plt.ylabel(column)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_EstadoFruta_Banano.png"),  bbox_inches='tight')
        #plt.show()

        # plt.figure(figsize=(10, 7))
        # sns.violinplot(data=d34, palette=colors)
        # plt.title(f'Violin Plot de {column} por Estado de la fruta (Banano)')
        # plt.ylabel(column)
        # plt.xticks(ticks=[0, 1], labels=['Fresco', 'Podrido'])
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_EstadoFruta_Banano.png"), bbox_inches='tight')
        # #plt.show()

        # #Estado de la fruta: Mango fresco vs podrido
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])

        bp = ax.boxplot(d56, patch_artist=True)
        #colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticklabels(['Fresco', 'Podrido'])
        plt.title(f'Box Plot de {column} por Estado de la fruta (Mango)')
        plt.ylabel(column)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_EstadoFruta_Mango.png"),  bbox_inches='tight')
        # #plt.show()

        # plt.figure(figsize=(10, 7))
        # sns.violinplot(data=d56, palette=colors)
        # plt.title(f'Violin Plot de {column} por Estado de la fruta (Mango)')
        # plt.ylabel(column)
        # plt.xticks(ticks=[0, 1], labels=['Fresco', 'Podrido'])
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_EstadoFruta_Mango.png"), bbox_inches='tight')
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
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                rgb, mask, alpha = aplicar_en_mascara(img)
                features_vector_img = []
                features_vector_img.append(fruta)
                features_vector_img.append(estado)
                features_vector_img.append(f"{fruta}_{estado}")
                features_vector_img.extend(caracteristicas_color_hsv(rgb, mask, alpha, fruta_estado=f"{fruta}_{estado}"))
                features_vector_img.extend(caracteristicas_textura_GLCM(rgb, mask))
                features_vector_img.extend(caracteristicas_textura_Haralick(rgb, mask))


                #feature_vector_img = np.concatenate([[fruta],[estado], features_vector_img])
                feature_vectors.append(features_vector_img)
                #labels.append(f"{fruta}_{estado}")
                print(src)

    #labels = np.array(labels)
    #features_matrix = np.array(feature_vectors)
    features_matrix = pd.DataFrame(feature_vectors, columns=["fruit","state","fruit_state", "dominant_hue_1", "dominant_hue_2", 
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
    features_matrix["fruit_state"] = le.fit_transform(features_matrix["fruit_state"])

    print(features_matrix.head())
    features_matrix.to_csv(os.path.join(out_dir, "MatrizCaracteristicas.csv"), index=False)
    
    diagramas(base_in, "./galeria_resultados/Diagramas_pruebas", features_matrix)
    feature_selection(out_dir, features_matrix)

    print("Extracción de características de color y textura de prueba completada") 
    print("Matriz de características:")

    #print(features_matrix)
    #print(labels)

def feature_selection(out_dir="./galeria_resultados", features_matrix=None):
    #X = features
    #Y = Labels

    X = features_matrix.drop(columns=["fruit", "state", "fruit_state","dominant_hue_1", "dominant_hue_2", "Saturation_1", "Saturation_2", "Value_1", "Value_2", "mean_contrast_GLCM", "mean_entropy_GLCM"])
    Y = features_matrix["fruit"]

    selector = SelectKBest(score_func=f_classif, k=5) #Número de características a seleccionar

    X_selected = selector.fit_transform(X, Y)

    selected_features = X.columns[selector.get_support()]
    f_scores = selector.scores_[selector.get_support()]


    #fs = ReliefF(n_neighbors=10, n_features_to_keep=5)
    #selected_features_relief = fs.fit_transform(X.values, Y.values)

    print("Características seleccionadas ANOVA:", selected_features)
    print("F-scores:", f_scores)

    #print("Características seleccionadas ReliefF:", selected_features_relief)


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
                    rgb, mask, alpha = aplicar_en_mascara(img)
                    features_vector_img = []
                    features_vector_img.append(fruta)
                    features_vector_img.append(estado)
                    features_vector_img.append(f"{fruta}_{estado}")
                    features_vector_img.extend(caracteristicas_color_hsv(rgb, mask, alpha, fruta_estado=f"{fruta}_{estado}"))
                    features_vector_img.extend(caracteristicas_textura_GLCM(rgb, mask))
                    features_vector_img.extend(caracteristicas_textura_Haralick(rgb, mask))

                    #feature_vector_img = np.concatenate([[fruta],[estado], features_vector_img])
                    feature_vectors.append(features_vector_img)

    #features_matrix = np.array(feature_vectors)
    features_matrix = pd.DataFrame(feature_vectors, columns=["fruit","state", "fruit_state","dominant_hue_1", "dominant_hue_2",
                                                             "Saturation_1", "Saturation_2", "Value_1", "Value_2",
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
    features_matrix["fruit_state"] = le.fit_transform(features_matrix["fruit_state"])

    #Normalizar características 
    minmax_scaler = MinMaxScaler()
    numeric_cols = features_matrix.columns.difference(
    ["fruit", "state", "fruit_state"], sort=False) # Excluir columnas de etiquetas
    features_matrix_norm = features_matrix.copy()
    features_matrix_norm[numeric_cols] = minmax_scaler.fit_transform(
    features_matrix[numeric_cols])

    print(features_matrix_norm.head())
    features_matrix_norm.to_csv(os.path.join(out_dir, "MatrizCaracteristicasNormalizada.csv"), index=False)
    diagramas(base_dir, out_dir, features_matrix_norm)

    feature_selection(out_dir, features_matrix_norm)

    print("Extracción de características de color y texture completada") 

