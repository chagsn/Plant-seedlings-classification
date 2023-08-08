def prediction_segmentation() :
    import streamlit as st

    from PIL import Image
    import cv2
    import os

    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import requests
    import pathlib
    import io

    from pathlib import Path

    from tensorflow.keras.models import load_model
    import tensorflow as tf
   

    @st.cache_resource
    def chargement_modele(filepath ) : 
        ##Chargement des modèle 

        Unet_filepath = Path(filepath)
        unet = load_model(Unet_filepath)
        unet.compile('adam','sparse_categorical_accuracy','accuracy')

        return  unet

    unet = chargement_modele("Unet_finalmodel")

    ##Chargement du dataFrame contenant les chemins des images de la base de donnée
    df = pd.read_csv(Path("V2_Plant_Seedlings_DataFrame.csv"))


    # Segmentation pas seuillage 
    def segmentation_seuillage (img_dossier,nom) : 
            
            segmentation_threshold = 118
            fig,ax = plt.subplots(1,3,figsize=(20,40),subplot_kw=dict(xticks=[], yticks=[]))
            plant = nom
            img_rgb = img_dossier
            img_rgb = cv2.resize(img_rgb,(224,224))
                # Définition du masque par seuillage basique
            img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
            img_lab = cv2.resize(img_lab,(224,224))
            _,mask = cv2.threshold(img_lab[:,:,1],segmentation_threshold,255,cv2.THRESH_BINARY_INV)
            # Elimination du bruit par ouverture puis fermeture
            kernel = np.ones((2,2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masked_img = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)

            # Affichage   
            ax[0].imshow(img_rgb) # de l'image original
            ax[0].set_title(plant,fontsize=16)
            ax[1].imshow(mask,cmap='gray') # du mask realiser par seuillage
            ax[1].set_title("Masque",fontsize=16)
            ax[2].imshow(masked_img) # de l'image segmentée
            ax[2].set_title("Image segmentée",fontsize=16)

            st.pyplot(fig)

    # Segmentation pas Deep Learning (Unet)
    def segmentation_Unet(img,img_rgb,species_donnee) : 

        fig,ax = plt.subplots(1,3,figsize=(20,40),subplot_kw=dict(xticks=[], yticks=[]))

        img_rgb = img_rgb
        img_rgb = cv2.resize(img_rgb,(224,224))

        # Prediction du mask par le modèle
        pred_mask = unet.predict(img, verbose = 0).argmax(axis=-1)[0]

        # Supperposition du mask sur l'image original pour obtenir l'image segmenter
        img_np = np.array(img.numpy()[0] * 255, dtype=np.uint8)
        segmented_img = cv2.bitwise_and(img_np, img_np, mask=np.uint8(pred_mask))

        #Affichage
        ax[0].imshow(img_rgb) # de l'image original
        ax[0].set_title(species_donnee,fontsize=16)
        ax[1].imshow(pred_mask,cmap='gray') #du masque predit par le modèle
        ax[1].set_title("Masque de Unet",fontsize=16)
        ax[2].imshow(segmented_img) # de l'image segmentée
        ax[2].set_title("Image segmentée \n par Unet",fontsize=16)
        
        st.pyplot(fig)

    ### Titre
    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'> Résultats de la Segmentation </h1>", unsafe_allow_html=True)

    # Image jeu de données
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Pour une image de la base de données :   </h1>", unsafe_allow_html=True)
    st.write("")

    # Segmentation sur image des données
    with st.form("Choix d'une image aléatoire de la base de données"):
        # Selectbox pour l'espece
        classe = np.append('select',df['species'].unique())
        option = st.selectbox(
            'Sélectionner une espèce : ', classe, index=0)
        
        # Selectbox pour la méthode de segmentation
        type_segmentation = st.selectbox(
        'Sélectionner un type de segmentation : ', ('all','segmentation par seuillage','segmentation par Deep Learning'))
        
        # Bouton prediction
        submitted = st.form_submit_button("Prédiction")
        if submitted:
            if option != 'select':
                # choix d'un index au hasard dans le dataframe en fonction de l'espece selectionner
                idx = np.random.choice(df[df.species == option].index, replace = False)
                filepath = df['filepath'][idx]

                # ouverture de l'image au norme RGB 
                img_donnee = np.array(Image.open(filepath))
                # nom de l'espace associé a l'image
                species_donnee = df['species'][idx]

                img_unet_donnee = tf.io.read_file(filepath)
                img_unet_donnee = tf.io.decode_jpeg(img_unet_donnee, channels=3)
                img_unet_donnee = tf.image.resize(img_unet_donnee, [224, 224])
                img_unet_donnee = tf.cast(img_unet_donnee, tf.float32)/255
                img_unet_donnee = tf.expand_dims(img_unet_donnee, 0)
                
                if type_segmentation == 'segmentation par seuillage' : 
                    segmentation_seuillage (img_donnee,species_donnee)

                elif type_segmentation == 'segmentation par Deep Learning' : 
                    segmentation_Unet(img_unet_donnee,img_donnee,species_donnee)
                
                else : 
                    segmentation_seuillage (img_donnee,species_donnee)
                    segmentation_Unet(img_unet_donnee,img_donnee,species_donnee)

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Pour une image à partir d\'un dossier :   </h1>", unsafe_allow_html=True)
    st.write("")

    with st.form("Choix de l'image dans un dossier :"):
        # Coix de l'image dans un dossier 
        uploaded_files = st.file_uploader("Choisir l'image", accept_multiple_files=True)
        
        # Selectbox pour la méthode de segmentation
        type_segmentation = st.selectbox(
        'Sélectionner un type de segmentation : ', ('all','segmentation par seuillage','segmentation par Deep Learning'))

        submitted = st.form_submit_button("Prédiction")
        if submitted:
            for uploaded_file in uploaded_files:
                species = os.path.splitext(uploaded_file.name)[0]
                st.write(species)
                img_dossier = np.array(Image.open(uploaded_file))
 
                img_unet_dossier = tf.image.resize(img_dossier, [224, 224])
                img_unet_dossier = tf.cast(img_unet_dossier, tf.float32) / 255.0
                img_unet_dossier = tf.expand_dims(img_unet_dossier, 0)
            
                if type_segmentation == 'segmentation par seuillage' : 
                    segmentation_seuillage (img_dossier,species)
                
                elif type_segmentation == 'segmentation par Deep Learning' : 
                    segmentation_Unet(img_unet_dossier,img_dossier,species)
                    
                else : 
                    segmentation_seuillage (img_dossier,species)
                    segmentation_Unet(img_unet_dossier,img_dossier,species)

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Pour une image d\'internet :   </h1>", unsafe_allow_html=True)
    st.write("")

    with st.form("Choix de l'image sur internet :"):
        image_url = st.text_input("Entrer l'URL de l'image")

        type_segmentation = st.selectbox(
        'Sélectionner un type de segmentation : ', ('all','segmentation par seuillage','segmentation par Deep Learning'))
        
        submitted = st.form_submit_button("Prédiction")
        if submitted:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            img_internet = np.array(image)
            
            url_path = pathlib.Path(image_url)
            filename = url_path.name

            img_unet_url = tf.image.resize(img_internet, [224, 224])
            img_unet_url = tf.cast(img_unet_url, tf.float32) / 255.0
            img_unet_url = tf.expand_dims(img_unet_url, 0)

            if type_segmentation == 'segmentation par seuillage' : 
                segmentation_seuillage(img_internet,filename)

            elif type_segmentation == 'segmentation par Deep Learning' : 
                segmentation_Unet(img_unet_url,img_internet,filename)
                
            else : 
                segmentation_seuillage(img_internet,filename)
                segmentation_Unet(img_unet_url,img_internet,filename)

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")