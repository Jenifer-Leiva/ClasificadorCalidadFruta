from core.libs import ctk, Image
from tkinter import filedialog, messagebox

def crear_frame_principal(app, mostrar_resultado):
    ruta_imagen = None

    # =================================================================================================
    # FUNCION PARA CARGAR IMAGEN
    # =================================================================================================

    def seleccionar_imagen():
        nonlocal ruta_imagen

        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.bmp")
            ]
        )

        if ruta:
            print("Imagen seleccionada:", ruta)

            # Abrir imagen
            imagen = Image.open(ruta)

            # Redimensionar
            nueva_img = ctk.CTkImage(
                light_image=imagen,
                size=(350, 350)
            )

            # Actualizar label
            label_img.configure(image=nueva_img)

            # IMPORTANTE:
            # guardar referencia para que no se borre
            label_img.image = nueva_img
        
        ruta_imagen = ruta

    def clasificar():
        if not ruta_imagen:
            messagebox.showwarning(
                title="Imagen no seleccionada",
                message="Por favor selecciona una imagen antes de clasificar."
            )
            return
        mostrar_resultado(ruta_imagen)

# =================================================================================================


#                                        FRAME PRINCIPAL


# =================================================================================================

    main_frame = ctk.CTkFrame(
        app,
        fg_color="transparent"
    )


    # Columnas responsivas
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)

    # ==================================================
    # COLUMNA IZQUIERDA (IMAGEN)
    # ==================================================

    frame_left = ctk.CTkFrame(
        main_frame,
        fg_color="transparent"
    )

    frame_left.grid(row=0, column=0, sticky="nsew", padx=20)

    # Imagen
    img = ctk.CTkImage(
        light_image=Image.open("images/imgFruitDefault.png"),
        size=(350, 350)
    )

    label_img = ctk.CTkLabel(
        frame_left,
        image=img,
        text=""
    )

    label_img.pack(expand=True)

    # ==================================================
    # COLUMNA DERECHA (TEXTOS Y BOTONES)
    # ==================================================

    frame_right = ctk.CTkFrame(
        main_frame,
        fg_color="transparent"
    )

    frame_right.grid(row=0, column=1, sticky="nsew", padx=20)

    # ------------------------
    # TITULO
    # ------------------------

    labelTitulo = ctk.CTkLabel(
        frame_right,
        text="CLASIFICADOR CALIDAD DE FRUTAS",
        font=("Impact", 32, "bold"),
        text_color="#FFFFFF",
        justify="left"
    )

    labelTitulo.pack(anchor="w", pady=(20, 10))

    # ------------------------
    # DESCRIPCIÓN
    # ------------------------

    labelContenido = ctk.CTkLabel(
        frame_right,
        text="Selecciona una fruta para clasificar su tipo y determinar su estado de madurez.",
        font=("Helvetica", 16),
        text_color="#FFFFFF",
        wraplength=400,
        justify="left"
    )

    labelContenido.pack(anchor="w", pady=(0, 40))

    # ==================================================
    # BOTONES
    # ==================================================

    boton_style = {
        "text_color": "#00193A",
        "font": ("Helvetica", 15),
        "corner_radius": 20,
        "width": 220,
        "height": 50,
        "fg_color": "#96A6BC",
        "hover_color": "#75869C"
    }

    botonImg = ctk.CTkButton(
        frame_right,
        text="Seleccionar Imagen",
        command=seleccionar_imagen,
        **boton_style
    )

    botonImg.pack(anchor="w", pady=10)



    # ==================================================
    # BOTON PRINCIPAL
    # ==================================================

    botonClasificar = ctk.CTkButton(
        frame_right,
        text="Clasificar",
        font=("Helvetica", 18, "bold"),
        corner_radius=20,
        width=220,
        height=55,
        command=clasificar
    )

    botonClasificar.pack(anchor="w", pady=40)

    # ==================================================
    # FOOTER
    # ==================================================

    labelPie = ctk.CTkLabel(
        app,
        text="Universidad Militar Nueva Granada | Ingeniería Multimedia | Reconocimiento de Patrones | 2026-1",
        font=("Helvetica", 12),
        text_color="#FFFFFF"
    )

    labelPie.pack(side="bottom", pady=10)

    return main_frame
