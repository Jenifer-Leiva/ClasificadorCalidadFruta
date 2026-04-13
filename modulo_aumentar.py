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

def añadir_ruido(img, sigma=15):
    ruido = np.random.normal(0, sigma, img.shape).astype(np.uint8)
    img_ruido = cv2.add(img, ruido)
    return img_ruido

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
    

def augmentar_dataset(base_dir="./DatasetFrutasSegmentadas", out_dir="./DatasetFrutasAumentadas", porcentaje=0.4):
    for fruta in os.listdir(base_dir):
        fruta_path = os.path.join(base_dir, fruta)
        if not os.path.isdir(fruta_path):
            continue

        for estado in os.listdir(fruta_path):
            estado_path = os.path.join(fruta_path, estado)
            if not os.path.isdir(estado_path):
                continue

            out_estado_path = os.path.join(out_dir, fruta, estado)
            os.makedirs(out_estado_path, exist_ok=True)

            files = [f for f in os.listdir(estado_path) if f.lower().endswith(".png")]
            n_original = len(files)
            n_aug = int(n_original * porcentaje)

            print(f"{fruta}/{estado}: {n_original} originales, generando {n_aug} augmentadas")

            # 1. Copiar originales primero
            for f in files:
                src = os.path.join(estado_path, f)
                dst = os.path.join(out_estado_path, f)
                img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    cv2.imwrite(dst, img)

            # 2. Generar augmentadas después
            for i in range(n_aug):
                f = random.choice(files)
                img_path = os.path.join(estado_path, f)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                metodo = random.choice(["rotar", "voltear", "brillo", "ruido"])
                if metodo == "rotar":
                    aug_img = rotar(img, angle=random.choice([15, 30, 45, 60]))

                elif metodo == "voltear":
                    aug_img = voltear(img, mode=random.choice(["horizontal", "vertical"]))

                elif metodo == "brillo":
                    aug_img = aplicar_en_mascara(img, lambda x: cambiar_brillo(x, factor=random.uniform(0.7, 1.5)))

                else:  # ruido
                    aug_img = aplicar_en_mascara(img, lambda x: añadir_ruido(x, sigma=random.randint(10, 30)))

                filename = f"{os.path.splitext(f)[0]}_aug{i}.png"
                out_path = os.path.join(out_estado_path, filename)
                cv2.imwrite(out_path, aug_img)

    print("Data augmentation completado en carpeta DatasetFrutasAumentadas.")

