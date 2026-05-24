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
from modulo_caracteristicas_forma import extraer_caracteristicas_forma
from modulo_dataset import cargar_dataset, normalizar
from modulo_segmentar import division_segmentacion, segmentar_guardar, recortar_fruta
from modulo_aumentar import augmentar_dataset
from modulo_dividir import dividir_guardar_multioutput
from modulo_caracteristicas_color_textura import extraer_caracteristicas_tipo_estado, extract_features_test
from modulo_modeloSVM import modelo_Completo


def main():
    
    # PREPARACION DE LOS DATOS

    # 1. Cargar dataset completo
    #x, y, class_names = cargar_dataset()

    # 2. Normalizar
    #x = normalizar(x)

    # 3. Dividir
    #dividir_guardar_multioutput(base_in="./DatasetFrutas", size=128, output_root="./DatasetFrutasDivididas")

    # Rutas base
    base_dividida = "./DatasetFrutasDivididas"
    #subsets_segmentar = ["train", "val", "test"]  # solo aplicar a estas
    subsets = ["train"]#, "val"] 

    # 3. Segmentar y guardar máscaras
    #for subset in subsets_segmentar:
    #    base_in = os.path.join(base_dividida, subset)
    #    base_out = os.path.join("./DatasetFrutasSegmentadas", subset)
    #    division_segmentacion(base_in=base_in, base_out=base_out, size=128)
    
    # 4. Data augmentation
    #for subset in subsets:
    #    base_dir = os.path.join("./DatasetFrutasSegmentadas", subset)
    #    out_dir = os.path.join("./DatasetFrutasAumentadas", subset)
    #    augmentar_dataset(base_dir=base_dir, out_dir=out_dir, porcentaje=0.4)

    
    # 6. Extracción de características de color y textura
    #for subset in subsets:
    #    base_dir = os.path.join("./DatasetFrutasAumentadas", subset)
    #    out_dir = os.path.join("./galeria_resultados", subset)
        #prueba_caracteristicas(base_in= base_dir, out_dir= out_dir)
        
    #    features_matrix_tipo_fruta_train, features_matrix_estado_fruta_train,  selected_features_tipo, selected_features_state = extraer_caracteristicas_tipo_estado(base_dir=base_dir, out_dir=out_dir)
    
    # 6. Extracción de características de forma
    #extraer_caracteristicas_forma(base_dir="./DatasetFrutasAumentadas", out_dir="./galeria_resultados")
    
    #features_matrix_tipo_fruta_test, features_matrix_estado_fruta_test = extract_features_test(base_dir="./DatasetFrutasSegmentadas/test", out_dir="./galeria_resultados/test", features_selected_fruit = selected_features_tipo, features_Selected_state = selected_features_state)
    
    modelo_Completo(base_dir_train="./DatasetFrutasSegmentadas/train", base_dir_test="./DatasetFrutasSegmentadas/test")


if __name__ == "__main__":
    main()


