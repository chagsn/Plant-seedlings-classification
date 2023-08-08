def segmentation() : 
    
    import streamlit as st

    from pathlib import Path

    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt

    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    from PIL import Image
    import cv2

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.callbacks import Callback
    from timeit import default_timer as timer
    from tensorflow.keras.utils import load_img 
    import os

    #importation de DataFrame 
    df = pd.read_csv("V2_Plant_Seedlings_DataFrame.csv")

    # Fonction retournant x indices d'images aléatoires par espèce pour un dataframe
    def indice(x, df, num_of_species=12):
        if num_of_species > 12:
            print("Il n'y a que 12 espèces.")
        elif x > min(df.species.value_counts()):
            print ("Le dataset ne contient que",df.species.value_counts().sort_values()[0], df.species.value_counts().sort_values().index[0],".")
        else:
            indices = []
        for s in np.random.choice(df.species.unique(), size=num_of_species, replace=False):
            for i in range(x):
                idx = np.random.choice(df[df.species == s].index, replace = False)
                indices.append(idx)
        return indices

    indices = indice(1,df,12)

    ## Titre 
    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'>Recherche sur la segmentation </h1>", unsafe_allow_html=True)


    ## Introduction
    st.write ("""
    La solution au problème de fuite de données(data leakage) consiste simplement à supprimer l'arrière-plan des images pour ne conserver que les parties de l'image correspondant à la plante: il nous faut donc segmenter les images, afin d'identifier 
    et extraire les pixels correspondant à la plante.
    Par ailleurs, la segmentation permet également de fournir en entrée des modèles de Computer Vision des images plus simples, où la forme des feuilles ressort immédiatement. Nous espérons donc une amélioration 
    des performances de classification en incluant cette étape de segmentation dans le pre-processing. """)

    st.write("")

    ## Mise en oeuvre
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Mise en oeuvre :   </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    Etant donné que nous ne disposons pas de données permettant une segmentation supervisée, et puisque sur nos images les plantes et l'arrière-plan semblent présenter des profils de couleur assez distincts, 
    nous avons mis en œuvre une segmentation simple par seuillage: les pixels correspondant à la plante sont identifiés en fonction de leur intensité dans un espace colorimétrique donné.
    Concrètement, la segmentation par seuillage consiste à produire un masque binaire à partir de l'image et d'une valeur seuil: les pixels dont la valeur est supérieure au seuil sont fixés à 0 (noir), et ceux dont 
    la valeur est inférieure au seuil sont fixés à 255 (blanc). Ce masque est ensuite appliqué à l'image pour ne conserver que les pixels de valeur inférieure au seuil.""")

    st.write("")
    st.write("")

    ##Choix de l'espace de couleur 
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Choix de l'espace de couleurs et du canal :   </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    Pour ce faire il nous faut choisir l'espace de couleurs dans lequel le seuillage sera appliqué, ainsi que la valeur du seuil. Nous avons donc analysé un certain nombre d'images choisies aléatoirement dans 3 espaces 
    colorimétriques: 

    •	L'espace RGB: Red, Green, Blue

    •	L'espace HSV: Hue, Saturation, Value

    •	L'espace LAB: Lightness (du noir au blanc) , A (du vert au rouge), B (du bleu au jaune)

    La figure ci-dessous présente, pour une image donnée choisie aléatoirement, les valeurs de chacun des 3 canaux de couleurs R, G et B dans l’espace RGB. L'objectif est ici de déterminer le canal permettant de 
    distinguer au mieux la plante de l'arrière-plan.
    """)


    # Analyse colorimétrique dans les différents espaces de couleurs
    # Choix aléatoire d'une image dans le DataFrame
    index = 1159 #np.random.choice(df.index)
    plant = df['species'].iloc[index] + '(' + df['image_name'].iloc[index].strip('.png') + ')'

    # Ouverture de l'image (initialement au format RGB) et conversion dans les espaces HSV et LAB
    img_rgb = np.array(Image.open(df['filepath'].iloc[index]))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)

    # Affichage des 3 canaux de l'image dans chacun des 3 espaces colorimétriques

    titles=[["Red","Green","Blue"],
            ["Hue","Saturation","Value"],
            ["\nLightness (from black to white)","\nA (from Green to Red)","\nB (from Blue to Yellow)"]]

    # Analyse dans l'espace RGB
    fig_rgb,ax_rgb = plt.subplots(1,3,figsize=(12,5),subplot_kw=dict(xticks=[], yticks=[]))
    fig_rgb.suptitle("Analyse par canal dans l'espace RGB", y=0.86, fontsize=12)
    for c in range(3):
        ax_rgb[c].imshow(img_rgb[:,:,c],cmap="RdYlBu")
        ax_rgb[c].set_title(plant + " - " + titles[0][c],fontsize=10)

    st.pyplot(fig_rgb)


    st.write("""
    Si pour nos yeux il est facile de détecter la plante par sa couleur verte, l'exemple ci-dessus montre que la distinction entre la plante et le fond de l'image à partir des valeurs RGB n'est pas aussi 
    aisée.
    En effet la plupart des cailloux du fond ont des valeurs similaires à la plante pour les 3 canaux individuels R, G et B.

    Essayons maintenant de décomposer l'image dans les espaces HSV et LAB:

    """)

    # Analyse dans l'espace HSV
    fig_hsv,ax_hsv = plt.subplots(1,3,figsize=(12,5),subplot_kw=dict(xticks=[], yticks=[]))
    fig_hsv.suptitle("Analyse par canal dans l'espace HSV", y=0.86, fontsize=12)
    for c in range(3):
        ax_hsv[c].imshow(img_hsv[:,:,c],cmap="RdYlBu")
        ax_hsv[c].set_title(plant + " - " + titles[1][c],fontsize=10)

    st.pyplot(fig_hsv)
        
    # Analyse dans l'espace LAB
    fig_lab,ax_lab = plt.subplots(1,3,figsize=(12,5),subplot_kw=dict(xticks=[], yticks=[]))
    fig_lab.suptitle("Analyse par canal dans l'espace LAB", y=0.88, fontsize=12)
    for c in range(3):
        ax_lab[c].imshow(img_lab[:,:,c],cmap="RdYlBu")
        ax_lab[c].set_title(plant + " - " + titles[2][c],fontsize=10)

    st.pyplot(fig_lab)


    st.write("""
    Cette fois-ci la plante se détache nettement de l'arrière-plan dans le canal A de l'espace LAB. Ce constat s'est répété pour les différentes images testées.
    Nous utiliserons donc le canal A de l'espace de couleurs LAB pour effectuer le seuillage.

    """)

    st.write("")
    st.write("")

    ## Choix du seuil de segmentation avec l'algorithme d'Otsu
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Choix du seuil de segmentation avec l'algorithme d'Otsu :   </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    Il reste maintenant à déterminer la valeur du seuil utilisée pour la segmentation: pour que l'image soit correctement segmentée, les pixels dont la valeur du canal A est inférieure à ce seuil doivent 
    correspondre au mieux à la plante, tandis que ceux dont la valeur est supérieure au seuil doivent appartenir à l'arrière-plan.

    Une première possibilité consiste à utiliser l'algorithme d'Otsu, qui calcule automatiquement pour chaque image la valeur seuil qui minimise la variance intra-classe. Cette méthode présente l'avantage de ne pas 
    utiliser d'hyperparamètre pour réaliser la segmentation, et d'adapter le seuil de segmentation aux différentes images. Malheureusement cette approche ne permet pas d'obtenir des résultats satisfaisants pour 
    l'ensemble des images: comme le montre la figure ci-dessous, certaines images sont correctement segmentées, mais pour d'autres le résultat est très décevant.

    """)

    def seuil_Otsu():
        nb_spec = df['species'].nunique()
        indices = indice(1,df)
        fig,ax = plt.subplots(nb_spec,4,figsize=(25,5*nb_spec),subplot_kw=dict(xticks=[], yticks=[]))
        for k,index in enumerate(indices):
            
            plant = df['species'].iloc[index] + ' (' + df['image_name'].iloc[index].strip('.png') + ')'
            img_rgb = np.array(Image.open(df['filepath'].iloc[index]))
            # Définition du masque par seuillage basique
            img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
            _,mask = cv2.threshold(img_lab[:,:,1],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                # Elimination du bruit par ouverture puis fermeture
            kernel = np.ones((2,2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masked_img = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)

            
            ax[k,0].imshow(img_rgb)
            ax[k,0].set_title(plant,fontsize=16,fontweight='bold')
            ax[k,1].imshow(img_lab[:,:,1],cmap="RdYlBu")
            ax[k,1].set_title("Canal A de l'espace LAB",fontsize=16,fontweight='bold')
            ax[k,2].imshow(mask,cmap='gray')
            ax[k,2].set_title("Masque",fontsize=16,fontweight='bold')
            ax[k,3].imshow(masked_img)
            ax[k,3].set_title("Image segmentée",fontsize=16,fontweight='bold')
            fig.subplots_adjust(wspace=0.1)
        st.pyplot(fig)


    if st.button('Reset',key="segmentation d'Otsu"):
        seuil_Otsu()
    else : 
        seuil_Otsu()


    st.write("""
    Le calcul du seuil par la méthode d'Otsu ne s'avérant pas satisfaisant, essayons de déterminer par nous-mêmes une valeur seuil applicable à l'ensemble des images. """)

    ## Choix du seuil de segmentation fixe 
    st.write("")
    st.write("")

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Détermination d'un seuil fixe de segmentation :   </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    En première approche, analysons à nouveau nos images dans l'espace LAB. La figure ci-dessous présente une série d'images choisies aléatoirement, ainsi que le canal A dans l'espace LAB correspondant, 
    et l'histogramme des valeurs du canal A.""")

    def seuil_histogramme() :
        # Choix aléatoire d'une image par espèce 
        nb_spec = df['species'].nunique()
        indices = indice(1,df)

        # Visualisation pour chaque image du canal A de l'image (espace de couleurs LAB), et de l'histogramme correspondant
        fig,ax = plt.subplots(nb_spec,3,figsize=(13,3*nb_spec))
        for k,index in enumerate(indices):
            img_rgb = np.array(Image.open(df['filepath'].iloc[index]))
            ax[k,0].imshow(img_rgb)
            ax[k,0].set_xticks([])
            ax[k,0].set_yticks([])
            ax[k,0].set_title(df['species'].iloc[index],fontsize=15)
            img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
            ax1 = ax[k,1].imshow(img_lab[:,:,1],cmap="RdYlBu",vmin=100,vmax=140)
            fig.colorbar(ax1,ax=ax[k,1],location='left')
            ax[k,1].set_xticks([])
            ax[k,1].set_yticks([])
            ax[k,2].hist(img_lab[:,:,1].flatten(),rwidth = 0.8,bins=30)
            ax[k,2].set_xlim(100,140)
        ax[0,1].set_title("Canal A de l'espace LAB",fontsize=15)
        ax[0,2].set_title("Histogramme du canal A",fontsize=15) 

        st.pyplot(fig)

    if st.button('Reset',key="segmentation histogramme"):
        seuil_histogramme()
        
    else : 
        seuil_histogramme()



    st.write("""
    Cette analyse nous permet de dégager une première estimation du seuil à la valeur de 120: on constate en effet que les pixels de l'arrière-plan ont des valeurs majoritairement supérieures à 120, 
    et inversement pour les pixels des plantes.

    Pour affiner le choix du seuil de segmentation, nous avons segmenté les images en testant différentes valeurs de seuil  116, 118 et 121 pour l'ensemble des espèces. 
    """)

    # Définition d'une fonction effectuant une segmentation par seuillage, suivie d'une ouverture/fermeture
    def threshold_segmentation(img_rgb,threshold):
        # Conversion dans l'espace LAB
        img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
        # Redimensionnement des images
        img_rgb = cv2.resize(img_rgb,(224,224))
        img_lab = cv2.resize(img_lab,(224,224))
        # Segmentation basique par seuillage
        _,mask = cv2.threshold(img_lab[:,:,1],threshold,255,cv2.THRESH_BINARY_INV)
        # Elimination du bruit par ouverture puis fermeture
        kernel = np.ones((2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masked_img = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)
        return masked_img

    def seuil_fixe () : 
        nb_spec = df['species'].nunique()
        indices = indice(1,df)
        thresholds = [116,118,121]

        fig,ax = plt.subplots(nb_spec,4,figsize=(13,4*nb_spec))
        for k,index in enumerate(indices):
            img_rgb = np.array(Image.open(df['filepath'].iloc[index]))
            ax[k,0].imshow(img_rgb)
            ax[k,0].set_xticks([])
            ax[k,0].set_yticks([])
            ax[k,0].set_title(df['species'].iloc[index],fontsize=16,fontweight='bold')
            for t, value in enumerate(thresholds): 
                segm_imgs = threshold_segmentation(img_rgb, value)
                ax[k,t+1].imshow(segm_imgs)
                ax[k,t+1].set_title("seuil ="+str(thresholds[t]),fontsize=16,fontweight='bold')
                ax[k,t+1].set_xticks([])
                ax[k,t+1].set_yticks([])
        st.pyplot(fig)

    if st.button('Reset',key="segmentation fixe"):
        seuil_fixe()
        
    else : 
        seuil_fixe()

    st.write("""
    La valeur de 118 est celle qui permet d'obtenir le meilleur compromis vis-à-vis des résultats de segmentation pour les différentes espèces. En effet, comme illustré sur les figures, pour les valeurs de seuil 
    supérieures à 118, certaines images segmentées sont "bruitées" par des pixels de l'arrière-plan ; tandis que les valeurs de seuil inférieures à 118 font perdre beaucoup de pixels correspondant aux plantes. 
    La valeur de seuil de 118 est donc celle que nous avons finalement retenue pour segmenter nos images.""")


    ### Resultat 
    st.write("")
    st.write("")
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Résultats :   </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    Pour limiter le "bruit" sur les images segmentées et ainsi améliorer les résultats de la segmentation, une étape supplémentaire d'ouverture-fermeture (transformation morphologique) est réalisée sur le masque 
    obtenu avant son application à l'image originale. Cette opération permet en particulier de supprimer les éventuels pixels de l'arrière-plan encore présents après la segmentation proprement dite.""")

    def seuil_118 () : 
            # Choix aléatoire d'une image par espèce
        nb_spec = df['species'].nunique()
        indices = indice(1,df)

        segmentation_threshold = 118
        fig,ax = plt.subplots(nb_spec,3,figsize=(15,5*nb_spec),subplot_kw=dict(xticks=[], yticks=[]))

        for k,index in enumerate(indices):
            plant = df['species'].iloc[index] + ' (' + df['image_name'].iloc[index].strip('.png') + ')'
            img_rgb = np.array(Image.open(df['filepath'].iloc[index]))
            # Définition du masque par seuillage basique
            img_lab = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2LAB)
            _,mask = cv2.threshold(img_lab[:,:,1],segmentation_threshold,255,cv2.THRESH_BINARY_INV)
            # Elimination du bruit par ouverture puis fermeture
            kernel = np.ones((2,2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masked_img = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)
            
            ax[k,0].imshow(img_rgb)
            ax[k,0].set_title(plant,fontsize=16)
            ax[k,1].imshow(mask,cmap='gray')
            ax[k,1].set_title("Masque",fontsize=16)
            ax[k,2].imshow(masked_img)
            ax[k,2].set_title("Image segmentée",fontsize=16)

        st.pyplot(fig)

    if st.button('Reset',key="segmentation 118"):
        seuil_118 ()
        
    else : 
        seuil_118 ()


    st.write("""
    Les résultats ci-dessus montrent que la segmentation n'est pas optimale pour toutes les espèces. Les espèces à feuilles fines en particulier, comme Black-grass, Common wheat et Loose Silky-bent, 
    semblent plus difficiles à segmenter, avec parfois une perte de données. Néanmoins le process retenu est celui qui permet d'obtenir le meilleur compromis pour l'ensemble des espèces.""")


    st.write("")

    ## Implémentation dans le modèle
    st.write("")
    st.write("")

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Implémentation dans le modèle :   </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    Une fois mis au point, ce process (segmentation + ouverture-fermeture) est inséré au sein de la fonction de pre-processing du générateur d'images. Les images fournies en entrée des modèles sont ainsi 
    d'abord transformées géométriquement pour l'augmentation des données, puis segmentées à la volée, avant d'être injectées dans l'algorithme de Deep Learning pour l'entraînement ou la prédiction.""")

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    
