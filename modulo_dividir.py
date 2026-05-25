from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os, cv2, json
import numpy as np
from modulo_dataset import cargar_dataset


def dividir_guardar_multioutput(base_in="./DatasetFrutas", size=128, output_root="./DatasetFrutasDivididas"):
    """
    Divide el dataset en train/test para multi-output.
    La validación se realizará posteriormente usando K-Fold.
    """

    # --------------------------------------------------
    # 1. CARGAR DATASET
    # --------------------------------------------------
    x, y, class_names = cargar_dataset(extract_path=base_in, size=size)

    # --------------------------------------------------
    # 2. SEPARAR ETIQUETAS
    # --------------------------------------------------
    tipo_labels = []
    estado_labels = []

    for label in y:
        fruta, estado = class_names[label]
        tipo_labels.append(fruta)
        estado_labels.append(estado)

    tipo_labels = np.array(tipo_labels)
    estado_labels = np.array(estado_labels)

    # --------------------------------------------------
    # 3. CODIFICAR ETIQUETAS
    # --------------------------------------------------
    le_tipo = LabelEncoder()
    le_estado = LabelEncoder()

    y_tipo = le_tipo.fit_transform(tipo_labels)
    y_estado = le_estado.fit_transform(estado_labels)

    # --------------------------------------------------
    # 4. ESTRATIFICACIÓN COMBINADA
    # --------------------------------------------------
    y_strat = np.array([f"{t}_{e}" for t, e in zip(y_tipo, y_estado)])

    # --------------------------------------------------
    # 5. DIVISIÓN TRAIN / TEST
    # --------------------------------------------------
    x_train, x_test, y_tipo_train, y_tipo_test, y_estado_train, y_estado_test = train_test_split(
        x, y_tipo, y_estado,
        test_size=0.30,
        stratify=y_strat,
        random_state=42
    )


    print("DIVISIÓN DE DATOS")
    print("Train:", x_train.shape)
    print("Test:", x_test.shape)

    # --------------------------------------------------
    # 6. FUNCIÓN PARA GUARDAR
    # --------------------------------------------------
    def dividir_guardar(x_split, y_tipo_split, y_estado_split, split_name):
        labels_info = []

        for i, (img_flat, tipo, estado) in enumerate(zip(x_split, y_tipo_split, y_estado_split)):
            img = img_flat.reshape(size, size, 3)

            # convertir a uint8
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)

            class_folder = f"{le_tipo.inverse_transform([tipo])[0]}_{le_estado.inverse_transform([estado])[0]}"
            img_dir = os.path.join(output_root, split_name, class_folder)
            os.makedirs(img_dir, exist_ok=True)

            filename = f"{class_folder}_{i}.jpg"
            cv2.imwrite(os.path.join(img_dir, filename), img_uint8)

            labels_info.append({
                "file": filename,
                "tipo": le_tipo.inverse_transform([tipo])[0],
                "estado": le_estado.inverse_transform([estado])[0]
            })

        # guardar etiquetas
        with open(os.path.join(output_root, split_name, "labels.json"), "w") as f:
            json.dump(labels_info, f, indent=4)

        print(f"{split_name} guardado correctamente")

    # --------------------------------------------------
    # 7. GUARDAR SPLITS
    # --------------------------------------------------
    dividir_guardar(x_train, y_tipo_train, y_estado_train, "train")
    dividir_guardar(x_test, y_tipo_test, y_estado_test, "test")

    return (
        x_train, y_tipo_train, y_estado_train,
        x_test, y_tipo_test, y_estado_test,
        le_tipo, le_estado
    )


from collections import Counter

def mostrar_distribucion(y_tipo, y_estado, le_tipo, le_estado, nombre):
    print(f"\n📊 Distribución en {nombre}")

    # Decodificar etiquetas
    tipo_dec = le_tipo.inverse_transform(y_tipo)
    estado_dec = le_estado.inverse_transform(y_estado)

    # Conteo por tipo
    conteo_tipo = Counter(tipo_dec)
    print("\nTipo de fruta:")
    for k, v in conteo_tipo.items():
        print(f"  {k}: {v}")

    # Conteo por estado
    conteo_estado = Counter(estado_dec)
    print("\nEstado de fruta:")
    for k, v in conteo_estado.items():
        print(f"  {k}: {v}")

    # Conteo combinado
    combinados = [f"{t}_{e}" for t, e in zip(tipo_dec, estado_dec)]
    conteo_comb = Counter(combinados)

    print("\nCombinación tipo_estado:")
    for k, v in conteo_comb.items():
        print(f"  {k}: {v}")
    
    mostrar_distribucion(y_tipo_train, y_estado_train, le_tipo, le_estado, "TRAIN")
    mostrar_distribucion(y_tipo_test, y_estado_test, le_tipo, le_estado, "TEST")

print("DIVISIÓN DE DATOS")
