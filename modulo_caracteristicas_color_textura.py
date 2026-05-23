import json
from PIL import features
from cv2 import kmeans
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.feature_extraction import image

from libs import os,cv2, np, plt, mh, sns,pd, stats, multipletests# Para correción
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
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

    # plt.tight_layout()
    # plt.savefig(os.path.join("./galeria_resultados/Diagramas_pruebas", f"segmentacion_{fruta_estado}.png"))  

    # plt.show()

    # #print(np.unique(segmented_values_H))
    # print("Tonos dominantes (Hue):", dominant_hues)
    # print("Tonos dominantes (HSV):", centers[two_main_clusters])

    return dominant_hues[0], dominant_hues[1], Saturations[0], Saturations[1], Values[0], Values[1]#, hist_h_norm_flat

#--------------------------------------------------------
# Características de textura: GLCM 
#--------------------------------------------------------

def caracteristicas_textura_GLCM_patch(img, mask):

#--------------------------------------------------------
    #Características de textura GLCM
#--------------------------------------------------------
    #Pasar a blanco y negro

    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image[~mask] = 0

    patches=image.extract_patches_2d(gray_image,(16,16))

    for patch in patches:
        glcm = graycomatrix(patch.astype(np.uint8), distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
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

        feature_vector_patch = [mean_contrast, mean_entropy, mean_energy, mean_homogeneity, mean_correlation]
        matrix_patches = np.column_stack((feature_vector_patch))

    feature_vector_GLCM = np.std(matrix_patches, axis=0)

    return feature_vector_GLCM

    """
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

    """

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
    
#-------------------------------------------------------
    #Características de textura Frecuencia: Discrete Wavelet Features (DWT)
#--------------------------------------------------------

def diagramas(base_in="./DatasetFrutasAumentadas", out_dir="./galeria_resultados", features_matrix=None, first_column_name="fruit"):

    #Visualización de características

    labels_diagrams = ["Banana", "Mango"] if first_column_name == "fruit" else ["Fresco", "Podrido"]

    label_0 = features_matrix[features_matrix[first_column_name] == 0]
    label_1 = features_matrix[features_matrix[first_column_name] == 1]

    for column in features_matrix.columns:  # Saltar las dos primeras columnas (fruit y state)
        if column == "fruit" or column == "state" or column == "fruit_state":
            continue

        d_1 = label_0[column]
        d_2 = label_1[column]

        d12 = [d_1, d_2] # Banano vs Mango || Fresco vs Podrido

        #Tipo de fruta || Estado de fruta
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])

        bp = ax.boxplot(d12, patch_artist=True)
        colors = ['lightyellow', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticklabels(labels_diagrams)
        plt.title(f'Box Plot de {column} por {first_column_name} de fruta')
        plt.ylabel(column)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"BoxPlot_{column}_{first_column_name}.png"), bbox_inches='tight')
        #plt.show()
        plt.close(fig)

        # plt.figure(figsize=(10, 7))
        # sns.violinplot(data=d12)
        # plt.title(f'Violin Plot de {column} por {first_column_name} de fruta')
        # plt.ylabel(column)
        # plt.xticks(ticks=[0, 1], labels= labels_diagrams)
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f"ViolinPlot_{column}_TipoEstadoFruta.png"))
        # #plt.show()

       
def prueba_caracteristicas(base_in, out_dir):

    feature_vectors = [] 
    labels = []

    # recorrer todas las carpetas de clases dentro de base_in
    for class_name in os.listdir(base_in):
        print(f"Procesando clase: {class_name}")
        class_path = os.path.join(base_in, class_name)
        if not os.path.isdir(class_path):
            continue

    # for fruta in ["Banana", "Mango"]:
    #     for estado in ["Fresh", "Rotten"]:
    #         estado_path = os.path.join(base_in, fruta, estado)
    #         if not os.path.isdir(estado_path):
    #             continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        random_img = random.choice(files)

        src = os.path.join(class_path, random_img)
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is not None:
            rgb, mask, alpha = aplicar_en_mascara(img)
            features_vector_img = []
            class_parts = class_name.split("_")
            fruta = class_parts[0]
            estado = class_parts[1]
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
    diagramas(base_in, out_dir, features_matrix_norm)

    feature_selection(out_dir, features_matrix_norm)

    print("Extracción de características de color y texture completada") 


    #print(features_matrix)
    #print(labels)


#Ttest, corrección FDR y matriz de correlación que descarta en base a p-value de ttest

def feature_selection(out_dir, features_matrix=None, first_column_name="fruit"):
    #X = features
    #Y = Labels

    #X = features_matrix.drop(columns=["dominant_hue_1", "dominant_hue_2", "Saturation_1", "Saturation_2", "Value_1", "Value_2", "mean_contrast_GLCM", "mean_entropy_GLCM"])
    
    features_selected_ttest = []
    dfSelectedFeaturesComplete = features_matrix.copy()
    
    Y = features_matrix[first_column_name]  # Usar la primera columna como etiqueta
    X = features_matrix.drop(columns = first_column_name)

    #T-test (Two Tailed)
    #Hay diferencia entre las medias de las dos clases

    groups_ = features_matrix.groupby(first_column_name)
    label_0 = groups_.get_group(0)
    label_1 = groups_.get_group(1)

    p_values = []
    p_labels = []

    print("Grupos evaluados")
    print(groups_.groups)
    
    alpha = 0.05
    bool_levene = False

    for feature in X.columns:

        #Levene test
        levene_stat, levene_p_value = stats.levene(label_0[feature], label_1[feature])

        if(levene_p_value > alpha): #Se asume que las clases tienen igual varianza
            bool_levene = True;
        #--------

        t_val, p_val = stats.ttest_ind(label_0[feature], label_1[feature], equal_var = bool_levene)

        p_values.append(p_val)
        p_labels.append(feature)

    i = 0

    for feature in p_labels:
        if(p_values[i] < alpha):
        #print(f"La característica {feature} proporciona una buena separabilidad entre clases.")
            features_selected_ttest.append({"feature": feature, "p_value": p_values[i]})
    else:
        print(f"La característica {feature} no proporciona una buena separabilidad entre clases.")
        dfSelectedFeaturesComplete.drop(feature, axis = 1, inplace = True)
        i = i+1

    df_features_selected_ttest = pd.DataFrame(features_selected_ttest)
    df_features_selected_ttest = df_features_selected_ttest.sort_values(by='p_value', ascending=True)
    df_features_selected_ttest.reset_index(drop=True, inplace=True)

    #Corrección FDR
    alpha_correction = 0.05
    rejected, q_values, _, _ = multipletests(df_features_selected_ttest['p_value'], alpha=alpha_correction, method='fdr_bh')
    df_features_selected_ttest['q_value'] = q_values
    df_features_selected_ttest['significant'] = rejected

    print("Seleccionadas segun FDR")
    print(df_features_selected_ttest)

    features_to_remove_FDR = df_features_selected_ttest.loc[
    ~df_features_selected_ttest['significant'],
    'feature']

    df_features_selected_ttest = df_features_selected_ttest[
    df_features_selected_ttest['significant'] == True]

    print("Seleccionadas segun FDR descartadas")
    print(df_features_selected_ttest)

    dfSelectedFeaturesComplete.drop(columns=features_to_remove_FDR, inplace=True)
    dfSelectedFeaturesComplete.head()

    #Matriz de correlación

    matrix_correlation = dfSelectedFeaturesComplete.corr(method = "pearson")
    #Revisar tanto correlaciones positivas como negativas
    matrix_abs = matrix_correlation.abs()

    pairs_correlated_features = []
    #Por simetria quedarse sin la diagonal y con el triangulo superior
    reduced_matrix = matrix_abs.where(np.triu(np.ones(matrix_abs.shape),k=1).astype(np.bool))

    plt.figure(figsize=(10, 7))
    sns.heatmap(reduced_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Mapa de Calor Matriz de Correlación {first_column_name}")
    path_mc1 = os.path.join(out_dir, f"MatrizCorrelacion{first_column_name}")
    plt.savefig(path_mc1)
    plt.show()

    df_features_selected_CorrTtest = df_features_selected_ttest.copy();

    for i in range(reduced_matrix.shape[0]):
        for j in range(i+1, reduced_matrix.shape[0]):
            if reduced_matrix .iloc[i,j] >= 0.8:
                label_i = reduced_matrix.index[i]
                label_j = reduced_matrix.index[j]
                pairs_correlated_features.append((label_i,label_j))
                print(label_i, label_j)

    for pair in pairs_correlated_features:
        #feature1_row = df_features_selected_ttest.loc[pair[0]]
        feature1_row = df_features_selected_ttest[df_features_selected_ttest["feature"] == pair[0]]
        #feature2_row = df_features_selected_ttest.loc[pair[1]]
        feature2_row = df_features_selected_ttest[df_features_selected_ttest["feature"] == pair[1]]

        #Se eliminan las caracteristicas con mayor p-value de cada par

        if (feature1_row["p_value"].iloc[0] <= feature2_row["p_value"].iloc[0]):
            df_features_selected_CorrTtest = df_features_selected_CorrTtest[df_features_selected_CorrTtest["feature"] != pair[1]]
            if pair[1] in dfSelectedFeaturesComplete.columns:
                dfSelectedFeaturesComplete.drop(pair[1], axis = 1, inplace = True)
        else:
            df_features_selected_CorrTtest = df_features_selected_CorrTtest[df_features_selected_CorrTtest["feature"] != pair[0]]
            if pair[0] in dfSelectedFeaturesComplete.columns:
                dfSelectedFeaturesComplete.drop(pair[0], axis = 1, inplace = True)

    df_features_selected_CorrTtest.reset_index(drop=True, inplace=True)

    print("Nuevo df para correlationMatrix")
    print(dfSelectedFeaturesComplete.head())

    #dfSelectedFeaturesComplete.head()

    dfSelectedFeaturesComplete.to_csv(os.path.join(out_dir, f"MatrizCaracteristicasSeleccion_{first_column_name}.csv"), index=False)

    #Matriz de correlacion actualizada
    matrix_correlation_2 = dfSelectedFeaturesComplete.corr(method = "pearson")
    matrix_abs_2 = matrix_correlation_2.abs()
    reduced_matrix_2 = matrix_abs_2.where(np.triu(np.ones(matrix_abs_2.shape),k=1).astype(np.bool_))

    plt.figure(figsize=(10, 7))
    sns.heatmap(reduced_matrix_2, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Mapa de Calor Matriz de Correlación Actualizada por P-valor {first_column_name}")
    path_mc2 = os.path.join(out_dir, f"MatrizCorrelacionTtest_{first_column_name}")
    plt.savefig(path_mc2)
    plt.show()

    return dfSelectedFeaturesComplete

def extraer_caracteristicas_tipo_estado(base_dir, out_dir):
    feature_vectors = []

    #json_path = os.path.join(base_dir, "labels.json")
    #img_dir = os.path.join(base_dir, "images")

    #Leer json
    #with open(json_path, "r") as f:
    #    labels_ = json.load(f)

    #json a diccionario

    #labelsImg_dict = {
    #    f["file"]: {
    #        "tipo": f["tipo"],
    #        "estado": f["estado"]
    #    }
    #    for f in labels_
    #}

    # recorrer todas las carpetas de clases dentro de base_in
    for class_name in os.listdir(base_dir):
        print(f"Procesando clase: {class_name}")
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

    #files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
        
        for f in files:
            src = os.path.join(class_path, f)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                rgb, mask, alpha = aplicar_en_mascara(img)

                class_parts = class_name.split("_")
                fruta = class_parts[0]
                estado = class_parts[1]

                #fruta = labelsImg_dict[f]["tipo"]
                #estado = labelsImg_dict[f]["estado"]

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
                                                                "Information Measure of Correlation 2"  ])  

    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    le =LabelEncoder()
    features_matrix["fruit"] = le.fit_transform(features_matrix["fruit"])
    features_matrix["state"] = le.fit_transform(features_matrix["state"])
    features_matrix["fruit_state"] = le.fit_transform(features_matrix["fruit_state"])

    #Normalizar características 
    scaler = StandardScaler()
    numeric_cols = features_matrix.columns.difference(
    ["fruit", "state", "fruit_state"], sort=False) # Excluir columnas de etiquetas
    features_matrix_norm = features_matrix.copy()
    features_matrix_norm[numeric_cols] = scaler.fit_transform(
    features_matrix[numeric_cols])

    print(features_matrix_norm.head())

    os.makedirs(out_dir, exist_ok=True)
    features_matrix_norm.to_csv(os.path.join(out_dir, "MatrizCaracteristicasNormalizada.csv"), index=False)

    #Matrices especificas para cada tipo de fruta y estado

    features_matrix_tipo_fruta = features_matrix_norm.copy()
    features_matrix_tipo_fruta = features_matrix_tipo_fruta.drop(columns=['state', 'fruit_state'])
    #features_matrix_tipo_fruta.to_csv(os.path.join(out_dir, "MatrizCaracteristicasNormalizada_TipoFruta.csv"), index=False)

    features_matrix_estado_fruta = features_matrix_norm.copy()
    features_matrix_estado_fruta = features_matrix_estado_fruta.drop(columns=['fruit', 'fruit_state'])
    #features_matrix_estado_fruta.to_csv(os.path.join(out_dir, "MatrizCaracteristicasNormalizada_EstadoFruta.csv"), index=False)

    #Solo se grafican los boxplot de las features seleccionadas

    #Diagramas y feature selection para tipo de fruta
    features_matrix_tipo_fruta = feature_selection(out_dir, features_matrix_tipo_fruta, first_column_name="fruit")
    diagramas(base_dir, out_dir, features_matrix_tipo_fruta, first_column_name="fruit")


    #Diagramas y feature selection para estado de fruta
    features_matrix_estado_fruta = feature_selection(out_dir, features_matrix_estado_fruta, first_column_name="state")
    diagramas(base_dir, out_dir, features_matrix_estado_fruta, first_column_name="state")

    print("Extracción de características de color y textura completada") 
