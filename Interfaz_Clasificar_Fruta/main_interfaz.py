import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from core.libs import ctk
from framePrincipal import crear_frame_principal
from frameResult import crear_frame_resultado
from ProcesamientoImagen import ProcesarImagen, PrediccionModelo

ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.geometry("1000x600")
app.title("Clasificador de Calidad de Frutas")
app.configure(fg_color="#00193A")


# ==================================================
# FUNCIONES CAMBIO PANTALLA
# ==================================================

def ocultar_todos():
    main_frame.pack_forget()
    main_frame2.pack_forget()


def mostrar_inicio():
    ocultar_todos()
    main_frame.pack(fill="both", expand=True, padx=40, pady=40)
    # Inicio: mostrar frame principal


def mostrar_resultado(ruta_imagen=None):
    ocultar_todos()
    main_frame2.pack(fill="both", expand=True, padx=40, pady=40)

    # Si se proporciona una ruta de imagen, procesarla y actualizar la vista
    if ruta_imagen:
        try:
            print(f"\n🔄 Procesando imagen: {ruta_imagen}")
            
            # Procesar imagen y extraer características
            f_fruit, f_state = ProcesarImagen(ruta_imagen)
            print(f"✓ Características extraídas")
            
            # Realizar predicción
            pred_fruit, probs_fruit, pred_state, probs_state = PrediccionModelo(f_fruit, f_state)
            print(f"✓ Predicción completada")
            print(f"  - Predicción fruta: {pred_fruit}")
            print(f"  - Predicción estado: {pred_state}")
            print(f"  - Probabilidades fruta shape: {probs_fruit.shape}")
            print(f"  - Probabilidades estado shape: {probs_state.shape}")
            
            # Extraer valores escalares correctamente
            pred_fruit_str = str(pred_fruit[0]) if hasattr(pred_fruit, '__len__') else str(pred_fruit)
            pred_state_str = str(pred_state[0]) if hasattr(pred_state, '__len__') else str(pred_state)
            
            # probs_fruit[0] es el array de probabilidades, probs_state[0] es el array de probabilidades
            actualizar_resultado(
                ruta_imagen, 
                probs_fruit[0],      # Array [prob_banana, prob_mango]
                probs_state[0],      # Array [prob_fresco, prob_podrido]
                pred_fruit_str, 
                pred_state_str
            )
            print("✓ UI actualizada\n")
            
        except Exception as e:
            print(f"\n❌ Error al procesar la imagen: {e}")
            import traceback
            traceback.print_exc()
            print()




# ==================================================
# CREAR FRAMES SOLO UNA VEZ
# ==================================================

main_frame = crear_frame_principal(
    app,
    mostrar_resultado
)

main_frame2, actualizar_resultado = crear_frame_resultado(
    app,
    mostrar_resultado,
    mostrar_inicio
)


# ==================================================
# INICIO
# ==================================================

mostrar_inicio()

app.mainloop()