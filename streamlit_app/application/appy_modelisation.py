def modelisation() : 
    # importation des modules 
    import streamlit as st 
    import numpy as np 
    import pandas as pd
    import itertools

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    from PIL import Image
    from sklearn.preprocessing import LabelEncoder
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    import streamlit.components.v1 as components
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import itertools
    

    import seaborn as sns
    sns.set_theme(style="whitegrid")
    from pathlib import Path
    import cv2
    from skimage.transform import resize
    
## Titre 
    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'> Modélisation </h1>", unsafe_allow_html=True)

    
## Transfer Learning
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Modèles de Transfer Learning :  </h1>", unsafe_allow_html=True)
    
    with st.expander("Explication :"):
        st.write("""
        Le Transfer Learning désigne l’ensemble des méthodes qui permettent de transférer les connaissances acquises de modèles pré-entraînés 
        comme point de départ pour un nouveau modèle. Il permet de développer rapidement des modèles performants et résoudre efficacement des problèmes complexes en Computer Vision 
        notamment.

        Les modèles ont été pré-entrainés des semaines durant sur un jeu de données d'aujourd’hui très célèbre: ImageNet.""")
    
## Liste déroulante des structures des modèles
    st.markdown("<h1 style='font-size: 20px; color : red;'> Les modèles utilisés :  </h1>", unsafe_allow_html=True)

    
    st.write("""   
    •   VGG19 (2014)
                 
    •   MobileNetV2 (2018)
    
    •   ResNet50 (2015)
    
    """)
    
    st.write(" ")


    images = {
    'VGG19': 'vgg19_structure.png',
    'MobileNetV2': 'mobilenetv2_structure.png',
    'ResNet50': 'resnet50_structure.png'}

    # Sélection de l'image
    selected_image = st.selectbox('Structure du modèle', ["select",'VGG19','MobileNetV2','ResNet50'])

    # Chargement et affichage de l'image sélectionnée
    if selected_image != 'select' :
        image_path = images[selected_image]
        image = Image.open(image_path)
        st.image(image, caption=selected_image)

## Pre-processing initial
    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Pre-processing initial et augmentation des données :</strong></h1>', unsafe_allow_html=True)

    st.write("Paramètres du générateur")

    image_param_initial = plt.imread("generateur_initial.png")
    new_size = (309,300)  
    resized_img = resize(image_param_initial, new_size)
    st.image(resized_img)

# Résultats sur la précision et la prédiction
    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Résultats du Pre-processing initial et augmentation des données :</strong></h1>', unsafe_allow_html=True)

 # Charger les résultats dans un DF
    def create_dataframe(filename, model_name, data_augmentation, segmentation):
        df = pd.read_csv(filename, index_col=0)
        df = df.reset_index(names="Dataset")
        df = df.drop("loss", axis=1)
        df = df.rename({'accuracy': 'Accuracy'}, axis=1)
        df["Model"] = model_name
        df["Data augmentation"] = data_augmentation
        df["Segmentation"] = segmentation
        return df

    models_list = [
        ["lenet_strongaugm_nosegmentation_results.csv", "LeNet", "Augmentation forte", "Sans segmentation"],
        ["lenet_lowaugm_segmentation_results.csv", "LeNet", "Augmentation faible", "Avec segmentation"],
        ["vgg19_unfreezed_strongaugm_nosegmentation_results.csv", "VGG19-Unfreezed", "Augmentation forte", "Sans segmentation"],
        ["resnet50_freezed_strongaugm_nosegmentation_results.csv", "ResNet50-Freezed", "Augmentation forte", "Sans segmentation"],
        ["resnet50_freezed_lowaugm_nosegmentation_results.csv", "ResNet50-Freezed", "Augmentation faible", "Sans segmentation"],
        ["resnet50_freezed_lowaugm_segmentation_results.csv", "ResNet50-Freezed", "Augmentation faible", "Avec segmentation"],
        ["resnet50_unfreezed_lowaugm_segmentation_results.csv", "ResNet50-Unfreezed", "Augmentation faible", "Avec segmentation"],
        ["resnet50_unfreezed_lowaugm_nosegmentation_results.csv", "ResNet50-Unfreezed", "Augmentation faible", "Sans segmentation"],
        ["resnet50_unfreezed_strongaugm_nosegmentation_results.csv", "ResNet50-Unfreezed", "Augmentation forte", "Sans segmentation"],
        ["mobilenetV2_freezed_strongaugm_nosegmentation_results.csv", "MobileNetV2-Freezed", "Augmentation forte", "Sans segmentation"],
        ["mobilenetV2_freezed_lowaugm_nosegmentation_results.csv", "MobileNetV2-Freezed", "Augmentation faible", "Sans segmentation"],
        ["mobilenetV2_freezed_lowaugm_segmentation_results.csv", "MobileNetV2-Freezed", "Augmentation faible", "Avec segmentation"],
        ["mobilenetV2_unfreezed_strongaugm_nosegmentation_results.csv", "MobileNetV2-Unfreezed", "Augmentation forte", "Sans segmentation"],
        ["mobilenetV2_unfreezed_lowgaugm_segmentation_results.csv", "MobileNetV2-Unfreezed", "Augmentation faible", "Avec segmentation"],
        ["mobilenetV2_unfreezed_lowaugm_nosegmentation_results.csv", "MobileNetV2-Unfreezed", "Augmentation faible", "Sans segmentation"]
    ]
        

 # Initialisation du DataFrame contenant les résultats des différents modèles
    results = pd.DataFrame(columns=["Dataset", "Accuracy", "Model", "Data augmentation", "Segmentation"])

    for m in models_list:
        df = create_dataframe(*m)
        results = pd.concat(objs=[results, df], axis=0, ignore_index=True)

    df = results[(results["Model"].isin(values=["ResNet50-Freezed", "MobileNetV2-Freezed","ResNet50-Unfreezed","MobileNetV2-Unfreezed","VGG19-Unfreezed"])) & \
                 (results["Data augmentation"]=="Augmentation forte") & \
                 (results["Segmentation"]=="Sans segmentation")]
    

    fig, ax = plt.subplots()

    
    sns.barplot(data=df,x='Model',y='Accuracy',hue='Dataset',palette="viridis")
    plt.legend(loc='lower right',framealpha=1.)
    plt.xticks(range(5),["VGG19-Unfreezed","ResNet50-Freezed","ResNet50-Unfreezed","MobileNetV2-Freeze","MobileNetV2-Unfreezed"], fontsize=7)
    plt.ylim(0.5,1)
    plt.yticks(np.arange(0.5,1.05,0.05))
    plt.grid(visible = False) 
    for bars in ax.containers:
        ax.bar_label(bars,fmt='%.3f',fontsize=7)
    plt.title("Performance des premiers modèles implémentés",fontsize=12,pad=12)

    st.pyplot(fig)

    st.write("Rapport de classification eet la matrice de confusion du MobileNetV2 De-Freezé")
    
    image_rapport_classification_MobileNetV2_initial = "resultat_non_optimiser.png"
    st.image(image_rapport_classification_MobileNetV2_initial)

    st.write("Le VGG19 a été entrainé seulement avec l'augmentation forte. Nous ne l'avons pas gardé pour les étapes suivantes en raison de son temps d'entrainement très long pour une performance a priori moindre comparée à celle des autres modèles de Transfer Learning.")

## Grand Titre pour l'amélioration des modèles
    st.markdown("<h1 style='text-align: center; font-size: 40px; color: #00008b; text-decoration: underline;'> Pistes d'améliorations des modèles </h1>", unsafe_allow_html=True)

## Augmentation des données
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Augmentation des données :  </h1>", unsafe_allow_html=True)

    st.write("Difficultés soulevées par les paramètres initiaux d'augmentation des données :")
    st.write("Certaines images générées n'affichent qu'une partie de la plante, voire ne l'affichent pas du tout.")

    def indice(x, df, num_of_species=12):
        if num_of_species > 12:
            print("Il n'y a que 12 espèces.")
    
        elif x > min(df.species.value_counts()):
            print("Le dataset ne contient que ",df.species.value_counts().sort_values()[0], df.species.value_counts().sort_values().index[0], ".")
    
        else:
            indices = []
            for s in np.random.choice(df.species.unique(), size = num_of_species, replace=False):
                indices += list(np.random.choice(df[df.species == s].index, size = x, replace=False))
            return indices 

    df = pd.read_csv("V2_Plant_Seedlings_DataFrame.csv")

    df_train, df_test = train_test_split(df, train_size= 0.8, 
                                       shuffle= True, 
                                       random_state= 222, 
                                       stratify= df['species'])
    
    img_size = (224, 224)
    channels = 3
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)
    batch_size= 32

    train_data_generator = ImageDataGenerator(rescale = 1./255,
                                              horizontal_flip= True,
                                              vertical_flip = True,
                                              rotation_range=90,
                                              width_shift_range=0.3, 
                                              height_shift_range = 0.3,
                                              shear_range = 0.3, 
                                              zoom_range = 0.4,
                                              dtype='float32')
    
    #Affichage de l'augmmentation de données
    fig, ax = plt.subplots(12, 6, figsize=(20,48))
    for i, idx in enumerate(indice(1, df_train)):
        ax[i, 0].imshow(plt.imread(df_train.filepath[idx]))
        ax[i, 0].set_title(df_train.species[idx], fontstyle="italic", fontweight='bold', fontsize=9)
        ax[i, 0].axis("off")
        
        for j in range (1,6):
            ax[i, j].imshow(train_data_generator.random_transform(plt.imread(df_train.filepath[idx])))
            ax[i, j].set_title("Augmenté")
            ax[i, j].axis("off")
    
    st.pyplot(fig)

# Changement pre-processing
    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Atténuation des paramètres d\'augmentation de données</strong></h1>', unsafe_allow_html=True)

    # Chargement des images
    image1 = "Augmentation forte.png"
    image2 = "fleche.png"
    image3 = "Augmentation faible.png"

    # Création des colonnes
    col1, col2, col3 = st.columns(3)

    # Affichage des images dans les colonnes respectives
    with col1:
        st.image(image1, use_column_width=True)

    with col2:
        st.image(image2, use_column_width=True)
    
    with col3:
        st.image(image3, use_column_width=True)

## Impact de l'augmentation de données

    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Impact de l\'augmentation faible de données</strong></h1>', unsafe_allow_html=True)
    
    # Affichage graphique en barre

    df1 = results[(results["Model"].isin(values=["ResNet50-Freezed","MobileNetV2-Freezed"])) & \
                  (results["Segmentation"]=="Sans segmentation")].sort_values(by="Model",ascending=False)
    df2 = results[(results["Model"].isin(values=["ResNet50-Unfreezed","MobileNetV2-Unfreezed"])) & \
                  (results["Segmentation"]=="Sans segmentation")].sort_values(by="Model",ascending=False)
    df = pd.concat(objs=[df1,df2])

    # Impact sur le jeu test pour les deux modèles (freezés et de-freezés)

    fig, ax = plt.subplots()

    sns.barplot(data=df[df.Dataset == "test"], x='Model', y='Accuracy', hue='Data augmentation', palette="viridis")

    plt.legend(loc='lower right', framealpha=1.)
    plt.ylim(0.5, 1)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.xticks(rotation=15)
    plt.grid(visible = False) 

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f', fontsize=9)

    plt.title("Impact de l'atténuation de l'augmentation de données\n sur la précision des modèles sur le jeu test", fontsize=10)

    # Afficher le premier graphique
    st.pyplot(fig)
    
    # Observation
    st.write("""
    Observations :
        
    •  L'augmentation faible permet d'améliorer l'accuracy des modèles freezés
    
    •  L'augumentation faible ne permet pas d'améliorer l'accuracy des modèles de-freezés
    
    """)

    
## Segmentation des images

# Titre
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Segmentation des images :  </h1>", unsafe_allow_html=True)
    
    st.write("""
             
    En plus des observations issues de la data visualisation, nous avons utilisé la technique du Grad-CAM afin de mettre en évidence les régions importantes d'une image utilisée pour la classification. 
    
    Cette technique nous a permis d\'observer deux points importants :
        
    •	L'intégralité de la plante n'est pas toujours considérée.
        
    •	La plante est parfois occultée au profit d'un autre élément de l'image.
        
    """)
    
    image_Grad_CAM = "Grad-CAM.png"
    st.image(image_Grad_CAM) 
    
    st.write("Par conséquent, l'objectif de la segmentation des images est d'isoler la plante du fond dans le but d'éviter une éventuelle fuite de données par le biais des cailloux.")

# Paramètres de segmentation d'images

    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Etapes de segmentation optimale</strong></h1>', unsafe_allow_html=True)
    st.write("""
     
    •	Une segmentation par seuillage basée sur la couleur verte
    
    •	Une étape d’ouverture afin d’éliminer les faux positifs du masque. Il s’agit des artéfacts restés autour de la plante après segmentation
    
    •	Une étape de fermeture afin de restaurer les faux négatifs du masque : les pixels retirés qui étaient dans la plante après segmentation, ou plus vulgairement les « trous dans la plante »
    
    
    Les images ont toutes été redimensionnées à la même taille afin d’harmoniser le pre-processing.

 
    """)
    # Ajustement des paramètres d\'augmentation + Segmentation + Ouverture/Fermeture

    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Ajustement des paramètres d\'augmentation + Segmentation + Ouverture/Fermeture</strong></h1>', unsafe_allow_html=True)
    
    train_data_generator_lowaug = ImageDataGenerator(horizontal_flip= True,
                                              vertical_flip = True,
                                              rotation_range=90,
                                              width_shift_range=0.05,
                                              height_shift_range = 0.05,
                                              dtype='float32')
    def tsegm(input_img, threshold =  118):
    # Etape 1: Segmentation simple par seuillage
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
    
    # Affichage des images segmenter et avec l'augmentation faible des données
    fig, ax = plt.subplots(12, 6, figsize=(20, 48))
    for i, idx in enumerate(indice(1, df_train)):
        ax[i, 0].imshow(plt.imread(df_train.filepath[idx]))
        ax[i, 0].set_title(df_train.species[idx],fontsize=16, fontweight='bold')
        ax[i, 0].axis("off")
        for j in range (1,6):
            ax[i, j].imshow(tsegm(train_data_generator_lowaug.random_transform(plt.imread(df_train.filepath[idx])*255)))
            ax[i, j].set_title("Preprocessed",fontsize=16, fontweight='bold')
            ax[i, j].axis("off")
    st.pyplot(fig)
    
# Impact de la segmentation sur des images

    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Impact de la segmentation d\'images</strong></h1>', unsafe_allow_html=True)

    df1 = results[(results["Model"].isin(values=["ResNet50-Freezed","MobileNetV2-Freezed"])) & \
                  (results["Data augmentation"]=="Augmentation faible")].sort_values(by=["Model","Segmentation"],ascending=False)
    df2 = results[(results["Model"].isin(values=["ResNet50-Unfreezed","MobileNetV2-Unfreezed"])) & \
                  (results["Data augmentation"]=="Augmentation faible")].sort_values(by=["Model","Segmentation"],ascending=False)
    df = pd.concat(objs=[df1,df2])

    fig, ax = plt.subplots()

    sns.barplot(data=df[df.Dataset=="test"],x='Model',y='Accuracy',hue='Segmentation',palette="viridis")
    plt.legend(loc='lower right',framealpha=1.)
    plt.ylim(0.5,1)
    plt.yticks(np.arange(0.5,1.05,0.05))
    plt.xticks(rotation=15)
    plt.grid(visible = False) 
    for bars in ax.containers:
        ax.bar_label(bars,fmt='%.3f',fontsize=9)
    plt.title("Impact de la segmentation sur la précision des modèles sur le jeu test",fontsize=12);
        
    st.pyplot(fig)  

    st.write("""
    Observations :
        
    •  La segmentation d'images permet d'améliorer l'accuracy des modèles freezés.
    
    •  La segmentation d'images ne permet pas d'améliorer l'accuracy des modèles de-freezés.       
    
    """)
    
    
## Evaluation de l'impact global de l'augmentation faible des données et de la segmentation des images

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Comparaison des modèles améliorés avec les modèles initiaux : </h1>", unsafe_allow_html=True)

    st.write("Trois de nos modèles ont été intégralement ré-entraînés en cumulant ces deux améliorations : LeNet, ResNet50 et MobileNetV2 (avec de-freeze complet des modèles de base pour ResNet50 et MobileNetV2). ")


    
# Impact cumulé des deux axes d'améliorations 

    st.markdown('<h1 style="font-size: 20px; color : red;"><strong>Impact cumulé des deux axes d\'améliorations</strong></h1>', unsafe_allow_html=True)

    df1 = results[results["Model"]=="LeNet"].sort_values(by="Segmentation",ascending=False)
    df2 = results[(results["Model"].isin(values=["ResNet50-Freezed","MobileNetV2-Freezed"])) & \
                  (((results["Data augmentation"]=="Augmentation forte")&(results["Segmentation"]=="Sans segmentation")) | \
                  ((results["Data augmentation"]=="Augmentation faible")&(results["Segmentation"]=="Avec segmentation")))]
    df2 = df2.sort_values(by=["Model","Segmentation"],ascending=False)
    df3 = results[(results["Model"].isin(values=["ResNet50-Unfreezed","MobileNetV2-Unfreezed"])) & \
                  (((results["Data augmentation"]=="Augmentation forte")&(results["Segmentation"]=="Sans segmentation")) | \
                  ((results["Data augmentation"]=="Augmentation faible")&(results["Segmentation"]=="Avec segmentation")))]
    df3 = df3.sort_values(by=["Model","Segmentation"],ascending=False)
    df = pd.concat(objs=[df1,df2,df3])

    model_evolution = df["Segmentation"].map({"Sans segmentation":"Initial model","Avec segmentation":"Improved model"})
    df.insert(3,'Model evolution',model_evolution)
    
    fig, ax = plt.subplots()
    
    sns.barplot(data=df[df.Dataset=="test"],x='Model',y='Accuracy',hue='Model evolution',palette="viridis")
    plt.legend(loc='lower right',framealpha=1.)
    plt.ylim(0.5,1)
    plt.yticks(np.arange(0.5,1.05,0.05))
    plt.xticks(rotation=15)
    plt.grid(visible = False) 
    for bars in ax.containers:
        ax.bar_label(bars,fmt='%.3f',fontsize=9)
    plt.title("Impact cumulé de l'augmentation faible et de la segmentation\n sur la précision des modèles sur le jeu test",fontsize=12)
    
    st.pyplot(fig)
    
    st.write("""
    Conclusion :
        
    •  Ce nouveau pre-processing permet d'améliorer les résultats du modèle LeNet et des modèles freezés ResNet50 et MobileNetV2.
    
    •  Ce nouveau pre-processing ne permet pas d'améliorer l'accuracy des modèles de-freezés.        
    
    """)
        
    st.image('resultats_optimiser.png' )  

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")        
        