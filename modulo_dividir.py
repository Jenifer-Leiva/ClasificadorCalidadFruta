from sklearn.model_selection import train_test_split
import os, cv2
import numpy as np
from modulo_dataset import cargar_dataset

def dividir_guardar(base_in="./DatasetFrutas", size=128, output_root="./DatasetFrutasDivididas"):
    """
    Divide el dataset segmentado y aumentado en train/val/test y guarda cada split en carpetas.
    """
 # 1. Cargar dataset directamente desde carpeta
    x, y, class_names = cargar_dataset(extract_path=base_in, size=size)
    
    # 1. División estratificada
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("DIVISION DE DATOS")
    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    # 2. Nombres de clases (ejemplo: ("Banana","Fresh"))
    classes = [f"{fruta}_{estado}" for fruta, estado in class_names]

    # 3. Helper para guardar cada split
    def guardar_split(x_split, y_split, split_name):
        for i, (img_flat, label) in enumerate(zip(x_split, y_split)):
            class_name = classes[label]

            # reconstruir imagen desde array
            img = img_flat.reshape(size, size, 3)

            # si está normalizada (0-1), volver a 0-255
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)

            # ruta destino
            path = os.path.join(output_root, split_name, class_name)
            os.makedirs(path, exist_ok=True)

            # guardar archivo
            cv2.imwrite(os.path.join(path, f"img_{i}.jpg"), img_uint8)

        print(f"{split_name} guardado en {output_root}/{split_name}")

    # 4. Guardar cada conjunto
    guardar_split(x_train, y_train, "train")
    guardar_split(x_val, y_val, "val")
    guardar_split(x_test, y_test, "test")

    return (x_train, y_train, x_val, y_val, x_test, y_test, class_names)
