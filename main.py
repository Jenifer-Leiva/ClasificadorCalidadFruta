#--------------------------------------------------------
# TERMINAL
#--------------------------------------------------------
# 2. crear entorno virtual
# python -m venv venv

# 1. quitar restricciones
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. activar entorno virtual
# .\venv\Scripts\Activate.ps1

# 4. iseleccionar el intérprete de Python del entorno virtual en VSCode
# ctrl+shift+p select interpreter venv

# 4. instalar dependencias
# .\venv\Scripts\python.exe -m pip install -r requirements.txt

from libs import np
from libs import os
from modulo_dataset import cargar_dataset, normalizar
from modulo_segmentar import division_segmentacion, segmentar_guardar, recortar_fruta
from modulo_aumentar import augmentar_dataset
from modulo_dividir import dividir_guardar
from modulo_caracteristicas_color_textura import extraer_caracteristicas


def main():

    # PREPARACION DE LOS DATOS

    # 1. Cargar dataset completo
    #x, y, class_names = cargar_dataset()

    # 2. Normalizar
    #x = normalizar(x)

    #dividir
    #dividir_guardar(base_in="./DatasetFrutas", size=128, output_root="./DatasetFrutasDivididas")



    # Rutas base
    base_dividida = "./DatasetFrutasDivididas"
    subsets = ["train", "val"]  # solo aplicar a estas

    # 3. Segmentar y guardar máscaras
    #for subset in subsets:
    #    base_in = os.path.join(base_dividida, subset)
    #    base_out = os.path.join("./DatasetFrutasSegmentadas", subset)
    #    division_segmentacion(base_in=base_in, base_out=base_out, size=128)

    # 4. Data augmentation
    #for subset in subsets:
    #    base_dir = os.path.join("./DatasetFrutasSegmentadas", subset)
    #    out_dir = os.path.join("./DatasetFrutasAumentadas", subset)
    #    augmentar_dataset(base_dir=base_dir, out_dir=out_dir, porcentaje=0.4)


    # 6. Extracción de características de color y textura
    extraer_caracteristicas(base_dir="./DatasetFrutasSegmentadas", out_dir="./galeria_resultados")

if __name__ == "__main__":
    main()


