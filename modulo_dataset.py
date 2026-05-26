#--------------------------------------------------------
# CARGA DE DATASET
#--------------------------------------------------------
from core.libs import zipfile, np, cv2, os

size = 128;
def cargar_dataset(zip_path="DatasetFrutas.zip", extract_path="./DatasetFrutas", size=128):
    # Extraer ZIP solo si no existe el directorio de destino
    if os.path.isdir(extract_path):
        print(f"Dataset ya existe en {extract_path}. Se omite extracción.")
    elif os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extraído desde ZIP.")
    elif os.path.isdir(zip_path):
        extract_path = zip_path
        print(f"Usando dataset existente en carpeta: {zip_path}")
    else:
        raise FileNotFoundError(
            f"No se encontró el dataset. Buscado zip en '{zip_path}' y carpeta en '{extract_path}'."
        )

    print("DATASET")
    print("Contenido disponible:", os.listdir(extract_path))

    x, y = [], []
    label_map = {
        ("Banana", "Fresh"): 0,
        ("Banana", "Rotten"): 1,
        ("Mango", "Fresh"): 2,
        ("Mango", "Rotten"): 3
    }
    class_names = list(label_map.keys())

    for fruta in ["Banana", "Mango"]:
        for estado in ["Fresh", "Rotten"]:
            estado_path = os.path.join(extract_path, fruta, estado)
            if not os.path.isdir(estado_path):
                continue

            for f in os.listdir(estado_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(estado_path, f)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    img = cv2.resize(img, (size, size))
                    x.append(img.flatten())
                    y.append(label_map[(fruta, estado)])

    return np.array(x), np.array(y), class_names


#--------------------------------------------------------
# Preprocesamiento / Normalizacion
#--------------------------------------------------------

def normalizar(x):
    return x.astype(np.float32) / 255.0


       


   
