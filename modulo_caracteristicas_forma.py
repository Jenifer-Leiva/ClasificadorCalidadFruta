import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =========================================================
# EXTRACCIÓN DE CARACTERÍSTICAS DE FORMA
# =========================================================

def analizar_forma(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

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

    return {

        "area": area,
        "perimetro": perimeter,
        "circularidad": circularidad,
        "compacidad": compacidad,
        "excentricidad": excentricidad,

        "hu1": hu_moments[0],
        "hu2": hu_moments[1],
        "hu3": hu_moments[2],
        "hu4": hu_moments[3],
        "hu5": hu_moments[4],
        "hu6": hu_moments[5],
        "hu7": hu_moments[6],
    }


# =========================================================
# FEATURE SELECTION
# =========================================================

def feature_selection_forma(
    out_dir,
    features_matrix,
    first_column_name="clase"
):

    df_selected = features_matrix.copy()

    # -----------------------------------------------------
    # T-TEST
    # -----------------------------------------------------

    groups_ = features_matrix.groupby(first_column_name)

    label_0 = groups_.get_group(0)
    label_1 = groups_.get_group(1)

    X = features_matrix.drop(columns=[first_column_name])

    p_values = []
    p_labels = []

    alpha = 0.05

    for feature in X.columns:

        levene_stat, levene_p = stats.levene(
            label_0[feature],
            label_1[feature]
        )

        equal_var = levene_p > alpha

        t_val, p_val = stats.ttest_ind(
            label_0[feature],
            label_1[feature],
            equal_var=equal_var
        )

        p_values.append(p_val)
        p_labels.append(feature)

    # -----------------------------------------------------
    # FEATURES SELECCIONADAS
    # -----------------------------------------------------

    features_selected = []

    for i, feature in enumerate(p_labels):

        if p_values[i] < alpha:

            features_selected.append({
                "feature": feature,
                "p_value": p_values[i]
            })

        else:

            print(f"❌ Eliminada: {feature}")

            if feature in df_selected.columns:
                df_selected.drop(feature, axis=1, inplace=True)

    df_selected_features = pd.DataFrame(features_selected)

    df_selected_features = df_selected_features.sort_values(
        by="p_value",
        ascending=True
    )

    df_selected_features.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------
    # FDR
    # -----------------------------------------------------

    rejected, q_values, _, _ = multipletests(
        df_selected_features["p_value"],
        alpha=0.05,
        method="fdr_bh"
    )

    df_selected_features["q_value"] = q_values
    df_selected_features["significant"] = rejected

    print("\n✅ FEATURES SIGNIFICATIVAS")
    print(df_selected_features)

    # -----------------------------------------------------
    # ELIMINAR FEATURES NO SIGNIFICATIVAS
    # -----------------------------------------------------

    remove_fdr = df_selected_features.loc[
        ~df_selected_features["significant"],
        "feature"
    ]

    df_selected.drop(columns=remove_fdr, inplace=True)

    # -----------------------------------------------------
    # MATRIZ DE CORRELACIÓN
    # -----------------------------------------------------

    matrix_corr = df_selected.corr(method="pearson")

    matrix_abs = matrix_corr.abs()

    reduced_matrix = matrix_abs.where(
        np.triu(
            np.ones(matrix_abs.shape),
            k=1
        ).astype(np.bool_)
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        reduced_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        f"Matriz de Correlación - {first_column_name}"
    )

    path_corr = os.path.join(
        out_dir,
        f"MatrizCorrelacion_{first_column_name}.png"
    )

    plt.savefig(path_corr, bbox_inches="tight")
    plt.show()

    plt.close()

    print(f"✅ Heatmap guardado en: {path_corr}")

    # -----------------------------------------------------
    # ELIMINAR FEATURES ALTAMENTE CORRELACIONADAS
    # -----------------------------------------------------

    correlated_pairs = []

    for i in range(reduced_matrix.shape[0]):

        for j in range(i + 1, reduced_matrix.shape[1]):

            if reduced_matrix.iloc[i, j] >= 0.8:

                f1 = reduced_matrix.index[i]
                f2 = reduced_matrix.columns[j]

                correlated_pairs.append((f1, f2))

    for pair in correlated_pairs:

        feature1 = pair[0]
        feature2 = pair[1]

        row1 = df_selected_features[
            df_selected_features["feature"] == feature1
        ]

        row2 = df_selected_features[
            df_selected_features["feature"] == feature2
        ]

        if row1.empty or row2.empty:
            continue

        p1 = row1["p_value"].iloc[0]
        p2 = row2["p_value"].iloc[0]

        # Eliminar la de peor p-value

        if p1 <= p2:

            if feature2 in df_selected.columns:
                df_selected.drop(feature2, axis=1, inplace=True)

        else:

            if feature1 in df_selected.columns:
                df_selected.drop(feature1, axis=1, inplace=True)

    # -----------------------------------------------------
    # MATRIZ FINAL
    # -----------------------------------------------------

    matrix_corr_2 = df_selected.corr(method="pearson")

    matrix_abs_2 = matrix_corr_2.abs()

    reduced_matrix_2 = matrix_abs_2.where(
        np.triu(
            np.ones(matrix_abs_2.shape),
            k=1
        ).astype(np.bool_)
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        reduced_matrix_2,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        f"Matriz Correlación Final - {first_column_name}"
    )

    path_corr_2 = os.path.join(
        out_dir,
        f"MatrizCorrelacionFinal_{first_column_name}.png"
    )

    plt.savefig(path_corr_2, bbox_inches="tight")
    plt.show()

    plt.close()

    # -----------------------------------------------------
    # GUARDAR CSV FINAL
    # -----------------------------------------------------

    csv_path = os.path.join(
        out_dir,
        f"MatrizCaracteristicasSeleccionadas_{first_column_name}.csv"
    )

    df_selected.to_csv(csv_path, index=False)

    print(f"✅ CSV guardado en: {csv_path}")

    return df_selected


# =========================================================
# PIPELINE COMPLETO
# =========================================================

def extraer_caracteristicas_forma(
    base_dir="./DatasetFrutasAumentadas",
    out_dir="./galeria_resultados"
):

    out_forma_dir = os.path.join(out_dir, "forma")

    os.makedirs(out_forma_dir, exist_ok=True)

    resultados = []

    # -----------------------------------------------------
    # RECORRER DATASET
    # -----------------------------------------------------

    for subset in ["train", "val"]:

        subset_path = os.path.join(base_dir, subset)

        if not os.path.isdir(subset_path):
            continue

        for class_name in os.listdir(subset_path):

            class_path = os.path.join(
                subset_path,
                class_name
            )

            if not os.path.isdir(class_path):
                continue

            for f in os.listdir(class_path):

                if not f.lower().endswith(".png"):
                    continue

                img_path = os.path.join(class_path, f)

                metrics = analizar_forma(img_path)

                if metrics is None:
                    continue

                metrics["subset"] = subset
                metrics["clase"] = class_name

                resultados.append(metrics)

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    df = pd.DataFrame(resultados)

    # -----------------------------------------------------
    # LABEL ENCODER
    # -----------------------------------------------------

    le = LabelEncoder()

    df["clase"] = le.fit_transform(df["clase"])

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    features = [
        "area",
        "perimetro",
        "circularidad",
        "compacidad",
        "excentricidad",
        "hu1",
        "hu2",
        "hu3",
        "hu4",
        "hu5",
        "hu6",
        "hu7"
    ]

    # -----------------------------------------------------
    # NORMALIZACIÓN
    # -----------------------------------------------------

    scaler = StandardScaler()

    df[features] = scaler.fit_transform(df[features])

    # -----------------------------------------------------
    # CSV NORMALIZADO
    # -----------------------------------------------------

    csv_norm = os.path.join(
        out_forma_dir,
        "MatrizCaracteristicasForma_Normalizada.csv"
    )

    df.to_csv(csv_norm, index=False)

    # -----------------------------------------------------
    # FEATURE SELECTION
    # -----------------------------------------------------

    df_selection = df[["clase"] + features]

    df_selection = feature_selection_forma(
        out_forma_dir,
        df_selection,
        first_column_name="clase"
    )

    # -----------------------------------------------------
    # BOXPLOTS
    # -----------------------------------------------------

    selected_features = [
        c for c in df_selection.columns
        if c != "clase"
    ]

    for feat in selected_features:

        plt.figure(figsize=(8, 6))

        sns.boxplot(
            x="clase",
            y=feat,
            data=df,
            hue="subset"
        )

        sns.stripplot(
            x="clase",
            y=feat,
            data=df,
            hue="subset",
            dodge=True,
            alpha=0.3,
            color="black"
        )

        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(
            handles[:2],
            labels[:2],
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        plt.title(f"Distribución de {feat}")

        if feat.startswith("hu"):
            plt.ylim(-10, 10)

        plt.tight_layout()

        out_file = os.path.join(
            out_forma_dir,
            f"{feat}_boxplot.png"
        )

        plt.savefig(out_file)

        plt.close()

    print("\n✅ EXTRACCIÓN DE FORMA COMPLETADA")