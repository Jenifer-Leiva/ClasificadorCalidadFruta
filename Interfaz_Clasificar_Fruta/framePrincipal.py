from core.libs import ctk, Image

def crear_frame_principal(app, mostrar_resultado):

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
        **boton_style
    )

    botonImg.pack(anchor="w", pady=10)

    botonGaleria = ctk.CTkButton(
        frame_right,
        text="Seleccionar Galería",
        **boton_style
    )

    botonGaleria.pack(anchor="w", pady=10)

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
        command=mostrar_resultado
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
