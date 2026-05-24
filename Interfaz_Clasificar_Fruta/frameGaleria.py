from core.libs import ctk, Image

def crear_frame_galeria(app, mostrar_resultado, volver_inicio):
# =================================================================================================


#                                        FRAME GALERIA


# =================================================================================================

    main_frame3 = ctk.CTkFrame(
        app,
        fg_color="transparent"
    )


    # ==================================================
    # TITULO
    # ==================================================

    labelTitulo3 = ctk.CTkLabel(
        main_frame3,
        text="GALERÍA DE FRUTAS",
        font=("Impact", 32, "bold"),
        text_color="#FFFFFF"
    )

    labelTitulo3.pack(anchor="w", pady=(20, 10))

    # ==================================================
    # DESCRIPCIÓN
    # ==================================================

    labelContenido3 = ctk.CTkLabel(
        main_frame3,
        text="Selecciona filtros para visualizar las frutas clasificadas.",
        font=("Helvetica", 16),
        text_color="#FFFFFF"
    )

    labelContenido3.pack(anchor="w", pady=(0, 30))

    # ==================================================
    # CONTENEDOR FILTROS
    # ==================================================

    filters_container = ctk.CTkFrame(
        main_frame3,
        fg_color="transparent"
    )

    filters_container.pack(fill="x", pady=(0, 30))

    # ==================================================
    # CONTENEDOR GALERIA
    # ==================================================

    gallery_container = ctk.CTkScrollableFrame(
        main_frame3,
        fg_color="transparent",
        width=900,
        height=350
    )

    gallery_container.pack(fill="both", expand=True)

    # ==================================================
    # DATOS EJEMPLO
    # ==================================================

    imagenes_data = []

    # 12 imágenes de ejemplo
    for i in range(12):

        if i % 4 == 0:
            clase = "Banano/Sano"

        elif i % 4 == 1:
            clase = "Banano/Podrido"

        elif i % 4 == 2:
            clase = "Mango/Sano"

        else:
            clase = "Mango/Podrido"

        imagenes_data.append({
            "image": "images/imgFruitDefault.png",
            "label": clase
        })

    # ==================================================
    # FUNCIÓN CREAR GALERÍA
    # ==================================================

    def cargar_galeria(filtro=None):

        # LIMPIAR GALERÍA
        for widget in gallery_container.winfo_children():
            widget.destroy()

        columnas = 3

        fila = 0
        columna = 0

        for item in imagenes_data:

            texto = item["label"]

            # FILTRADO
            if filtro is not None:
                if filtro.lower() not in texto.lower():
                    continue

            # ==================================================
            # CARD
            # ==================================================

            card = ctk.CTkFrame(
                gallery_container,
                fg_color="#0D274F",
                corner_radius=20,
                width=220,
                height=260
            )

            card.grid(
                row=fila,
                column=columna,
                padx=15,
                pady=15,
                sticky="n"
            )

            # ==================================================
            # IMAGEN
            # ==================================================

            img = ctk.CTkImage(
                light_image=Image.open(item["image"]),
                size=(180, 180)
            )

            label_img = ctk.CTkLabel(
                card,
                image=img,
                text=""
            )

            label_img.pack(pady=(15, 10))

            # IMPORTANTE
            label_img.image = img

            # ==================================================
            # TEXTO
            # ==================================================

            label_text = ctk.CTkLabel(
                card,
                text=texto,
                font=("Helvetica", 16, "bold"),
                text_color="#FFFFFF"
            )

            label_text.pack(pady=(0, 15))

            # ==================================================
            # ORGANIZAR GRID
            # ==================================================

            columna += 1

            if columna >= columnas:
                columna = 0
                fila += 1

    # ==================================================
    # FUNCIONES FILTRO
    # ==================================================

    def mostrar_todos():
        cargar_galeria()

    def filtrar_banano():
        cargar_galeria("Banano")

    def filtrar_mango():
        cargar_galeria("Mango")

    def filtrar_sano():
        cargar_galeria("Sano")

    def filtrar_podrido():
        cargar_galeria("Podrido")

    # ==================================================
    # BOTONES FILTRO
    # ==================================================

    filter_style = {
        "font": ("Helvetica", 14, "bold"),
        "corner_radius": 20,
        "height": 40
    }

    btnTodos = ctk.CTkButton(
        filters_container,
        text="Todos",
        fg_color="#96A6BC",
        text_color="#00193A",
        command=mostrar_todos,
        **filter_style
    )

    btnBanano = ctk.CTkButton(
        filters_container,
        text="Banano",
        fg_color="#FFD268",
        text_color="#84661F",
        command=filtrar_banano,
        **filter_style
    )

    btnMango = ctk.CTkButton(
        filters_container,
        text="Mango",
        fg_color="#BCFB7C",
        text_color="#52851E",
        command=filtrar_mango,
        **filter_style
    )

    btnSano = ctk.CTkButton(
        filters_container,
        text="Sano",
        fg_color="#9CE6FF",
        text_color="#12546A",
        command=filtrar_sano,
        **filter_style
    )

    btnPodrido = ctk.CTkButton(
        filters_container,
        text="Podrido",
        fg_color="#83A7B3",
        text_color="#06303E",
        command=filtrar_podrido,
        **filter_style
    )

    # ==================================================
    # PACK FILTROS
    # ==================================================

    btnTodos.pack(side="left", padx=5)
    btnBanano.pack(side="left", padx=5)
    btnMango.pack(side="left", padx=5)
    btnSano.pack(side="left", padx=5)
    btnPodrido.pack(side="left", padx=5)

    # ==================================================
    # BOTÓN VOLVER
    # ==================================================

    botonVolver2 = ctk.CTkButton(
        main_frame3,
        text="Volver",
        font=("Helvetica", 18, "bold"),
        corner_radius=20,
        width=220,
        height=55,
        command=volver_inicio
    )

    botonVolver2.pack(anchor="w", pady=40)

    # ==================================================
    # CARGAR GALERÍA INICIAL
    # ==================================================

    cargar_galeria()

    return main_frame3