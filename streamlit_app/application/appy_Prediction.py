def prediction() :  
    import streamlit as st 

    from pathlib import Path

    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt

    import itertools 

    from PIL import Image
    import cv2
    import os

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import load_img, img_to_array
    from tensorflow.keras.applications import mobilenet_v2, resnet50
    import requests
    import pathlib
    from io import BytesIO


    df = pd.read_csv(Path("V2_Plant_Seedlings_DataFrame.csv"))


    label2classes = {0: 'Black-grass',
                    1: 'Charlock',
                    2: 'Cleavers',
                    3: 'Common Chickweed',
                    4: 'Common wheat',
                    5: 'Fat Hen',
                    6: 'Loose Silky-bent',
                    7: 'Maize',
                    8: 'Scentless Mayweed',
                    9: "Shepherd's Purse",
                    10: 'Small-flowered Cranesbill',
                    11: 'Sugar beet'}


    @st.cache_resource
    def chargement_modele(mobilenet_filepath, resnet_filepath,vgg19_filepath) : 
        ##Chargement des modèle 
        # Modèle MobileNetV2 optimisé: avec augmentation faible + segmentation

        mobilenet_filepath = Path(mobilenet_filepath)
        mobilenet = load_model(mobilenet_filepath)
        mobilenet.compile('adam','sparse_categorical_accuracy','accuracy')

        # Modèle ResNet50 optimisé: avec augmentation faible + segmentation
        resnet_filepath = Path(resnet_filepath)
        resnet = load_model(resnet_filepath)
        resnet.compile('adam','sparse_categorical_accuracy','accuracy')

        # Modèle VGG19: avec augmentation forte, sans segmentation

        vgg19_filepath = Path("vgg19_strongaugm_nosegmentation_finalmodel",vgg19_filepath)
        vgg19 = load_model(vgg19_filepath,compile=False)
        vgg19.compile('adam','sparse_categorical_accuracy','accuracy')

        return  mobilenet,resnet,vgg19


    mobilenet,resnet,vgg19 = chargement_modele("mobilenetV2_lowaugm_segmentation_finalmodel","resnet50_lowaugm_segmentation_finalmodel","VGG19_9507.h5")

    # Fonction de segmentation par seuillage

    def threshold_segmentation(input_img):
        threshold=118
        img_rgb = np.uint8(input_img)
        img_rgb = cv2.resize(img_rgb,(224,224))
        img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
        
        # Création du masque par seuillage
        _,mask = cv2.threshold(img_lab[:,:,1],threshold,255,cv2.THRESH_BINARY_INV)
        
        # Elimination du bruit par ouverture puis fermeture
        kernel = np.ones((2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Application du masque à l'image
        masked_img = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)
        
        return masked_img

    # Fonctions de pre-processing incluant la segmentation par seuillage pour les modèles MobileNetV2 et ResNet50

    def mobilenet_preprocessing_with_tsegm(input_img):
        
        # Etape 1: Segmentation simple par seuillage
        masked_img = threshold_segmentation(input_img)

        # Etape 2: Pre-processing propre à MobileNetV2 (normalisation dans l'intervalle [-1,1]))
        final_input = mobilenet_v2.preprocess_input(masked_img)
        
        return final_input    

    def resnet_preprocessing_with_tsegm(input_img):
        
        # Etape 1: Segmentation simple par seuillage
        masked_img = threshold_segmentation(input_img)

        # Etape 2: Pre-processing propre à ResNet50 (normalisation dans l'intervalle [-1,1]))
        final_input = resnet50.preprocess_input(masked_img)
        
        return final_input    
    

    # Fonctions de pre-procesing + prédiction des différents modèles

    def mobilenet_preprocess_and_predict(img_filepath):
        
        """
        Fonction réalisant le pre-processing d'une image avec segmentation par seuillage, puis la classification 
        par le modèle MobileNetV2, et renvoyant les probabilités des différentes classes
        Argument:
        img_filepath: str: chemin d'accès à l'image
        """
        
        # Pre-processing
        
        segmented_img = mobilenet_preprocessing_with_tsegm(img_filepath)
        input_arr = np.array([segmented_img])  # Convert single image to a batch.
        
        # Prédiction
        pred = np.squeeze(mobilenet.predict(input_arr,verbose=0))
            
        return pred

    def resnet_preprocess_and_predict(img_filepath):
        
        """
        Fonction réalisant le pre-processing d'une image avec segmentation par seuillage, puis la classification 
        par le modèle ResNet50, et renvoyant les probabilités des différentes classes
        Argument:
        img_filepath: str: chemin d'accès à l'image
        """
        
        # Pre-processing
        segmented_img = resnet_preprocessing_with_tsegm(img_filepath)
        input_arr = np.array([segmented_img])  # Convert single image to a batch.
        
        # Prédiction
        pred = np.squeeze(resnet.predict(input_arr,verbose=0))
            
        return pred

    def vgg19_preprocess_and_predict(img_filepath):
        
        """
        Fonction réalisant le pre-processing d'une image (sans segmentation), puis la classification 
        par le modèle VGG19, et renvoyant les probabilités des différentes classes
        Argument:
        img_filepath: str: chemin d'accès à l'image
        """
        
        # Pre-processing
        img = img_to_array(img_filepath)/255.
        input_arr = np.array([img])  # Convert single image to a batch.
        
        # Prédiction
        pred = np.squeeze(vgg19.predict(input_arr,verbose=0))
            
        return pred

    # Comparaison des prédictions des modèles VGG19, MobileNetV2, ResNet50
    def show_model_pred(img_filepath,model='MobileNetV2'):
        
        """
        Fonction affichant pour une image et un modèle donnés les 3 classes les plus probables prédites par le modèle.
        Arguments:
        img_filepath: str: chemin d'accès à l'image
        model: str: Modèle CNN utilisé pour la prédiction: au choix MobileNetV2', 'VGG19', 'ResNet50' ou 'all' pour comparer les 
                    3 modèles
        """
        
        # Définition d'une fonction retournant un DataFrame des 3 classes les plus problables avec leurs probabilités
        # à partir du np array des prédictions d'un modèle
        
        def pred_to_df(pred):
            
            # Conversion des prédictions en DataFrame
            df = pd.Series(pred).reset_index().set_axis(['Classe','Probabilité'],axis=1)
            df['Classe'] = df['Classe'].map(label2classes)
        
            # Affichage des 3 classes les plus probables avec leurs probabilités
            df = df.sort_values(by='Probabilité',ascending=False).iloc[:3]
            df.index = (range(1,4))
            df['Probabilité'] = df['Probabilité'].map('{:.2%}'.format)
            
            return df
        
        
        # Affichage d'un DataFrame des prédictions pour un modèle donné
    
        if model != 'all':
            
            # Choix du modèle
            if model == 'MobileNetV2':
                st.write("Modèle MobileNetV2:")
                pred = mobilenet_preprocess_and_predict(img_filepath)
            elif model == 'VGG19':
                st.write("Modèle VGG19:")
                pred = vgg19_preprocess_and_predict(img_filepath)
            elif model == 'ResNet50':
                st.write("Modèle ResNet50:")
                pred = resnet_preprocess_and_predict(img_filepath)
            else:
                st.write("Modèle non reconnu.")
                return
        
            # Affichage du DataFrame des prédictions
            st.dataframe(pred_to_df(pred))
        
        
        # Comparaison des prédictions des 3 modèles dans un unique DataFrame
        
        else:
            
            df1 = pred_to_df(vgg19_preprocess_and_predict(img_filepath))
            st.write("Modèle VGG19:")
            st.dataframe(df1)
            df2 = pred_to_df(mobilenet_preprocess_and_predict(img_filepath))
            st.write("Modèle MobileNetV2:")
            st.dataframe(df2)
            df3 = pred_to_df(resnet_preprocess_and_predict(img_filepath))
            st.write("Modèle ResNet50:")
            st.dataframe(df3)
            


    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'>Démonstration des modèles</h1>", unsafe_allow_html=True)



    ##Choix de l'image
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Choix de l'image dans la base de données :   </h1>", unsafe_allow_html=True)
    st.write("")

    # Dans la base de données
    with st.form("Choix d'une image aléatoire de la base de données"):
        classe = np.append('select',df['species'].unique())
        option = st.selectbox(
            'Sélectionner une espèce : ', classe, index=0)
        
        modele_donnee = st.selectbox(
        'Sélectionner le modèle : ', ('all','VGG19','MobileNetV2','ResNet50'))

        submitted = st.form_submit_button("Prédiction")
        if submitted:
            if option != 'select':
                idx = np.random.choice(df[df.species == option].index, replace = False)
                filepath = df['filepath'][idx]
                img_donnee = load_img(filepath,target_size=(224,224))
                species_donnee = df['species'][idx]
                st.write(species_donnee)
                st.image(img_donnee)
                

            show_model_pred(img_donnee,modele_donnee)


    # Dans un dossier   
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Choix de l'image dans un dossier :   </h1>", unsafe_allow_html=True)
    st.write("")
        
    with st.form("Choix de l'image dans un dossier :"):
        uploaded_files = st.file_uploader("Choisir l'image", accept_multiple_files=True)

        modele_dossier = st.selectbox(
        'Sélectionner le modèle : ', ('all','VGG19','MobileNetV2','ResNet50'))

        submitted = st.form_submit_button("Prédiction")
        if submitted:
            for uploaded_file in uploaded_files:
                species = os.path.splitext(uploaded_file.name)[0]
                st.write(species)
                img_dossier = load_img(uploaded_file,target_size=(224,224))
                st.image(img_dossier)
                
                show_model_pred(img_dossier,modele_dossier)


    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Choix de l'image sur internet :   </h1>", unsafe_allow_html=True)
    st.write("")

    #URL Internet
    with st.form("Choix de l'image sur internet :"):
        image_url = st.text_input("Entrer l'URL de l'image")
        
        modele_internet = st.selectbox(
        'Sélectionner le modèle : ', ('all','VGG19','MobileNetV2','ResNet50'))
        
        submitted = st.form_submit_button("Prédiction")
        if submitted:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        
            img_internet = load_img(BytesIO(image_data), target_size=(224, 224))
            
            url_path = pathlib.Path(image_url)
            filename = url_path.name
            st.image(img_internet)
            st.write("Nom de l'image :", filename)

            show_model_pred(img_internet, modele_internet)
