from libs import os, cv2, np

# --- Funciones de augmentación ---
def rotar(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def voltear(img, mode="horizontal"):
    if mode == "horizontal":
        return cv2.flip(img, 1)
    elif mode == "vertical":
        return cv2.flip(img, 0)
    else:
        return img

def cambiar_brillo(img, factor=1.2):
    img_float = img.astype(np.float32)
    img_bright = np.clip(img_float * factor, 0, 255)
    return img_bright.astype(np.uint8)


# --- Pipeline ---
import random
from libs import os, cv2, np


def aplicar_en_mascara(img, funcion_aug):
    if img.shape[2] == 4:  # RGBA
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]

        mask = alpha > 0

        rgb_aug = rgb.copy()
        rgb_mod = funcion_aug(rgb)

        # Aplicar solo donde hay fruta
        rgb_aug[mask] = rgb_mod[mask]

        return np.dstack((rgb_aug, alpha))
    else:
        return funcion_aug(img)
    

def augmentar_dataset(base_dir, out_dir, porcentaje=0.4):
    """
    base_dir: carpeta de entrada (ej. ./DatasetFrutasSegmentadas/train)
    out_dir: carpeta de salida (ej. ./DatasetFrutasAumentadas/train)
    porcentaje: proporción de imágenes augmentadas respecto a las originales
    """

    if not os.path.isdir(base_dir):
        print(f"⚠️ No existe la carpeta {base_dir}, se omite.")
        return

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Crear carpeta destino
        out_class_path = os.path.join(out_dir, class_name)
        os.makedirs(out_class_path, exist_ok=True)

        files = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
        if not files:
            print(f"⚠️ No hay imágenes en {class_path}")
            continue

        n_original = len(files)
        n_aug = int(n_original * porcentaje)

        print(f"{base_dir}/{class_name}: {n_original} originales, generando {n_aug} augmentadas")

        # Copiar originales
        for f in files:
            src = os.path.join(class_path, f)
            dst = os.path.join(out_class_path, f)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                cv2.imwrite(dst, img)

        # Generar augmentadas
        for i in range(n_aug):
            f = random.choice(files)
            img_path = os.path.join(class_path, f)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            metodo = random.choice(["rotar", "voltear", "brillo", "ruido"])
            if metodo == "rotar":
                aug_img = rotar(img, angle=random.choice([15, 30, 45, 60]))
            elif metodo == "voltear":
                aug_img = voltear(img, mode=random.choice(["horizontal", "vertical"]))
            else:
                aug_img = aplicar_en_mascara(img, lambda x: cambiar_brillo(x, factor=random.uniform(0.7, 1.5)))
            

            filename = f"{os.path.splitext(f)[0]}_aug{i}.png"
            out_path = os.path.join(out_class_path, filename)
            cv2.imwrite(out_path, aug_img)

    print(f"✅ Data augmentation completado en {out_dir}")

