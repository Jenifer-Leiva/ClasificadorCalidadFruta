#--------------------------------------------------------
# CARGA DE DATASET
#--------------------------------------------------------
from libs import zipfile, np, cv2, os

size = 128;
def cargar_dataset(zip_path="DatasetFrutas.zip", extract_path="./DatasetFrutas", size=128):
    # Extraer ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("DATASET")
    print("Contenido extraído:", os.listdir(extract_path))

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


       


   
