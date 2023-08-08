def Unet() : 
    import streamlit as st 

    from pathlib import Path

    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools 

    from PIL import Image
    import cv2
    from skimage import io
    import os
    from tensorflow.keras.models import load_model

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D, concatenate
    from tensorflow.keras.optimizers import Adam

    from tensorflow.keras.callbacks import Callback
    from timeit import default_timer as timer

    def indice(x, df, num_of_species=12):
        
        """
        Fonction retournant x indices d'images aléatoires par espèce d'un dataframe
        """
        
        if num_of_species > 12:
            print("Il n'y a que 12 espèces.")
        
        elif x > min(df.species.value_counts()):
            print("Le dataset ne contient que ",df.species.value_counts().sort_values()[0], df.species.value_counts().sort_values().index[0], ".")
        
        else:
            indices = []
            for s in np.random.choice(df.species.unique(), size=num_of_species, replace=False):
                for i in range(x):
                    idx = np.random.choice(df[df.species == s].index, replace = False)
                    indices.append(idx)
            return indices
        

    @st.cache_resource
    def chargement_modele(filepath ) : 
        ##Chargement des modèle 

        Unet_filepath = Path(filepath)
        unet = load_model(Unet_filepath)
        unet.compile('adam','sparse_categorical_accuracy','accuracy')

        return  unet


    unet = chargement_modele("Unet_finalmodel")

    def threshold_segmentation(img_dossier) : 
        # Choix aléatoire d'une image par espèce
        
        segmentation_threshold = 118
        img_rgb = img_dossier
            # Définition du masque par seuillage basique
        img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
        _,mask = cv2.threshold(img_lab[:,:,1],segmentation_threshold,255,cv2.THRESH_BINARY_INV)
        # Elimination du bruit par ouverture puis fermeture
        kernel = np.ones((2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masked_img = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)

        return masked_img
                    
                
    df = pd.read_csv("V2_Plant_Seedlings_DataFrame.csv")

    def segmentation_Unet() : 
        nb_spec = df['species'].nunique()
        indices = indice(1,df)
        fig,ax = plt.subplots(nb_spec,4,figsize=(15,5*nb_spec),subplot_kw=dict(xticks=[], yticks=[]))

        for k,index in enumerate(indices):
            plant = df['species'].iloc[index] + ' (' + df['image_name'].iloc[index].strip('.png') + ')'
            img_rgb = np.array(Image.open(df['filepath'].iloc[index]))
            img = tf.io.read_file(df['filepath'].iloc[index])
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32)/255
            img = tf.expand_dims(img, 0)
            
            pred_mask = unet.predict(img, verbose = 0).argmax(axis=-1)[0]

            img_np = np.array(img.numpy()[0] * 255, dtype=np.uint8)
            segmented_img = cv2.bitwise_and(img_np, img_np, mask=np.uint8(pred_mask))
            segm_imgs = threshold_segmentation(img_rgb)


            ax[k,0].imshow(img_rgb)
            ax[k,0].set_title(plant,fontsize=16)
            ax[k,1].imshow(pred_mask,cmap='gray')
            ax[k,1].set_title("Masque de Unet",fontsize=16)
            ax[k,2].imshow(segmented_img)
            ax[k,2].set_title("Image segmentée \n par Unet",fontsize=16)
            ax[k,3].imshow(segm_imgs)
            ax[k,3].set_title("Image segmentée \n par seuillage",fontsize=16)

        st.pyplot(fig)

    ## Titre 
    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'> Segmentation avec UNet </h1>", unsafe_allow_html=True)

        ## Les données  
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'>Les données :  </h1>", unsafe_allow_html=True)

    df_annotation = pd.read_csv("Plant_Mask_DataFrame.csv")

    st.dataframe(df_annotation.head()) 

    indices = indice(1,df_annotation,num_of_species= 6)
    fig,ax = plt.subplots(6,2,figsize=(7,2*6),subplot_kw=dict(xticks=[], yticks=[]))

    for k,index in enumerate(indices):
        plant = df_annotation['species'].iloc[index] + ' (' + df_annotation['name'].iloc[index].strip('.png') + ')'
        img_rgb = np.array(Image.open(df_annotation['filepath'].iloc[index]))
        img_annotation = np.array(Image.open(df_annotation['annotation'].iloc[index]))

        ax[k,0].imshow(img_rgb)
        ax[k,0].set_title(plant,fontsize=7)
        ax[k,1].imshow(img_annotation,cmap='gray')
        ax[k,1].set_title("Masque annotation",fontsize=7)

    st.pyplot(fig)


    # Introduction et description du modèle UNet
    st.write ("""
            L’un des réseaux de neurones les plus utilisés pour la segmentation d’images est U-NET. 
            Il s’agit d’un modèle de réseau de neurones entièrement convolutif. 
            Ce modèle fut initialement développé par Olaf Ronneberger, Phillip Fischer, et Thomas Brox en 2015 pour la segmentation d’images médicales.
            
    """)

    st.write("")








    ## Architecture du modèle UNet
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Architecture du modèle UNet :  </h1>", unsafe_allow_html=True)

    st.write("")

    ### Image de l'architecture
    st.image("Unet_architecture.png", use_column_width=True)

    ### Structure
    st.write("")

    st.markdown("""L'architecture de ce modèle (en "U" d'où le U-Net) est composée :""")

    st.markdown(""" - D'un réseau de neurones encodeur aussi appelé chemin contractant.
    
    Il se compose traditionnellement de couches de convolution et de Max-Pooling.
    
    Son but est d'extraire des features et caractéristiques importantes de l'image tout en réduisant sa taille pour diminuer le nombre de paramètres du réseau.""")
        
    st.markdown("""- D'un réseau de neurones décodeur, aussi appelé chemin d'expansion.
    
    Symétrique à l'encodeur, il utilise des couches d'upsampling visant à augmenter la dimension, ainsi que des couches de convolution et de concaténation.
    
    Son objectif est de projeter les features apprises par l'encodeur sur un espace de pixels de plus grande résolution, soit la résolution de l'image initiale.


    Les liaisons entre l'encodeur/décodeur permettent d'une part de diminuer le vanishing Gradient, et d'autre part de mieux transiter les informations à différentes échelles.
    Le modèle cumule ainsi près d’1.9 millions de paramètres répartis en 39 couches. 
        """
    )
       
    ## Entrainement du modèle
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Entraînement du modèle : </h1>", unsafe_allow_html=True)    

    st.write("")

    st.image("Historique d'entrainement de l'Unet sur 1500 images augmentées.png", use_column_width=True)

    st.write("")

    st.markdown(""" La meilleure accuracy obtenue au cours de l'entraînement  sur l'échantillon test est de 0.97746.""")
        
        

    ## Résultat du modèle  
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Résultats du modèle :  </h1>", unsafe_allow_html=True)


    if st.button('Reset',key="resultat Unet"):
            segmentation_Unet()
            
    else : 
            segmentation_Unet()
