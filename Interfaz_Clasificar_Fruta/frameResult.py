
from core.libs import ctk, Image

def crear_frame_resultado(app, mostrar_resultado, volver_inicio):
# =================================================================================================


#                                      FRAME RESULTADO IMAGEN


# =================================================================================================

    main_frame2 = ctk.CTkFrame(
        app,
        fg_color="transparent"
    )


    # Columnas responsivas
    main_frame2.grid_columnconfigure(0, weight=1)
    main_frame2.grid_columnconfigure(1, weight=1)

    # ==================================================
    # COLUMNA IZQUIERDA (IMAGEN)
    # ==================================================

    frame_left2 = ctk.CTkFrame(
        main_frame2,
        fg_color="transparent"
    )

    frame_left2.grid(row=0, column=0, sticky="nsew", padx=20)

    # Imagen
    img2 = ctk.CTkImage(
        light_image=Image.open("images/imgFruitDefault.png"),
        size=(350, 350)
    )

    label_img2 = ctk.CTkLabel(
        frame_left2,
        image=img2,
        text=""
    )

    label_img2.pack(expand=True)

    # ==================================================
    # COLUMNA DERECHA (TEXTOS Y BOTONES)
    # ==================================================

    frame_right2 = ctk.CTkFrame(
        main_frame2,
        fg_color="transparent"
    )

    frame_right2.grid(row=0, column=1, sticky="nsew", padx=20)

    # ------------------------
    # TITULO
    # ------------------------

    labelTitulo2 = ctk.CTkLabel(
        frame_right2,
        text="CLASIFICADOR CALIDAD DE FRUTAS",
        font=("Impact", 32, "bold"),
        text_color="#FFFFFF",
        justify="left"
    )

    labelTitulo2.pack(anchor="w", pady=(20, 10))


    # ==================================================
    # CONTENIDO RESULTADO   
    # ==================================================

    # ------------------------
    # CONTENENDOR CARDS
    # ------------------------
    cards_container = ctk.CTkFrame(
        frame_right2,
        fg_color="transparent"
    )

    cards_container.pack(anchor="w", pady=20, fill="x")

    card_style = {
        "width": 220,
        "height": 55,
        "corner_radius": 20,
    }

    cardText_style = {
        "font": ("Helvetica", 16, "bold"),
    }

    # ------------------------
    # CONTENEDORES ETIQUETAS
    # ------------------------

    banana_container = ctk.CTkFrame(
        cards_container,
        fg_color="transparent"
    )
    banana_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    mango_container = ctk.CTkFrame(
        cards_container,
        fg_color="transparent"
    )
    mango_container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    sano_container = ctk.CTkFrame(
        cards_container,
        fg_color="transparent"
    )
    sano_container.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

    podrido_container = ctk.CTkFrame(
        cards_container,
        fg_color="transparent"
    )
    podrido_container.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")



    # CARDS POR TIPO ESTADO

    # banano
    cardBanana = ctk.CTkFrame(
        banana_container,
        fg_color="#FFD268",
        **card_style
    )
    cardBanana.pack(fill="x")

    labelBanana = ctk.CTkLabel(
        cardBanana,
        text="Banano",
        text_color="#84661F",
        **cardText_style
    )
    labelBanana.pack(expand=True, pady=10)

    labelPBanana = ctk.CTkLabel(
        banana_container,
        text="50%",
        text_color="#FFFFFF",
        font=("Impact", 20)
    )
    labelPBanana.pack(pady=(5,0))



    #Mango
    cardmango = ctk.CTkFrame(
        mango_container,
        fg_color="#BCFB7C",
        **card_style
    )
    cardmango.pack(fill="x")

    labelmango = ctk.CTkLabel(
        cardmango,
        text="Mango",
        text_color="#52851E",
        **cardText_style
    )
    labelmango.pack(expand=True, pady=10)

    labelPMango = ctk.CTkLabel(
        mango_container,
        text="50%",
        text_color="#BCFB7C",
        font=("Impact", 20)
    )
    labelPMango.pack(pady=(5,0))

    #Sano
    cardsano = ctk.CTkFrame(
        sano_container,
        fg_color="#9CE6FF",
        **card_style
    )
    cardsano.pack(fill="x")

    labelsano = ctk.CTkLabel(
        cardsano,
        text="Sano",
        text_color= "#12546A",
        **cardText_style
    )
    labelsano.pack(expand=True, pady=10)

    labelPSano = ctk.CTkLabel(
        sano_container,
        text="50%",
        text_color="#9CE6FF",
        font=("Impact", 20)
    )
    labelPSano.pack(pady=(5,0))

    #Podrido
    cardPodrido = ctk.CTkFrame(
        podrido_container,
        fg_color="#83A7B3",
        **card_style
    )
    cardPodrido.pack(fill="x")

    labelPodrido = ctk.CTkLabel(
        cardPodrido,
        text="Podrido",
        text_color= "#06303E",
        **cardText_style
    )
    labelPodrido.pack(expand=True, pady=10)

    labelPPodrido = ctk.CTkLabel(
        podrido_container,
        text="50%",
        text_color="#83A7B3",
        font=("Impact", 20)
    )
    labelPPodrido.pack(pady=(5,0))

    # CARDS EN PANTALLA
    cards_container.grid_columnconfigure(0, weight=1)
    cards_container.grid_columnconfigure(1, weight=1)

    banana_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    mango_container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    sano_container.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    podrido_container.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")


    # ==================================================
    # BOTON VOLVER
    # ==================================================

    botonVolver = ctk.CTkButton(
        frame_right2,
        text="Volver",
        font=("Helvetica", 18, "bold"),
        corner_radius=20,
        width=220,
        height=55,
        command=volver_inicio
    )

    botonVolver.pack(anchor="w", pady=40)

    return main_frame2