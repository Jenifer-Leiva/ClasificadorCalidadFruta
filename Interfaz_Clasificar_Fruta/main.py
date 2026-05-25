import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from core.libs import ctk
from framePrincipal import crear_frame_principal
from frameResult import crear_frame_resultado
from frameGaleria import crear_frame_galeria
from ProcesamientoImagen import Prueba

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
    main_frame3.pack_forget()


def mostrar_inicio():
    ocultar_todos()
    main_frame.pack(fill="both", expand=True, padx=40, pady=40)
    Prueba()


def mostrar_resultado():
    ocultar_todos()
    main_frame2.pack(fill="both", expand=True, padx=40, pady=40)


def mostrar_galeria():
    ocultar_todos()
    main_frame3.pack(fill="both", expand=True, padx=40, pady=40)


# ==================================================
# CREAR FRAMES SOLO UNA VEZ
# ==================================================

main_frame = crear_frame_principal(
    app,
    mostrar_resultado
)

main_frame2 = crear_frame_resultado(
    app,
    mostrar_resultado,
    mostrar_inicio
)

main_frame3 = crear_frame_galeria(
    app,
    mostrar_resultado,
    mostrar_inicio
)


# ==================================================
# INICIO
# ==================================================

mostrar_inicio()

app.mainloop()