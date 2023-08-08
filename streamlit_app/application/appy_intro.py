def Intro () : 
    import streamlit as st
    import pandas as pd
    import numpy as np
    from PIL import Image

    st.markdown("<h1 style='text-align: center; font-size: 48px; color : blue;text-decoration: underline;'> Contexte </h1>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h1 style='font-size: 30px; color : green; text-decoration: underline;'> Jeu de données :  </h1>", unsafe_allow_html=True)

    st.write("""Le jeu de données est constitué de photographies de plantes au stade de semis appartenant à __12 espèces__ avec :""")
    st.write("  - 5 539 photographies")
    st.write("  - En format PNG")
    st.write("  - Espace de couleur RGB")
    st.write("  - Réparti en 12 sous dossiers")

    st.write("")
    st.write("")

    tableau_espece = pd.DataFrame({'Espèce':['Common wheat','Maize','Sugar beet','Black-grass','Charlock','Cleavers','Common Chickweed','Fat hen','Loose Silky-bent','Scentless Mayweed','Sheperds Purse','Small-flowered Cranesbill'],
                                'Traduction' : ['Froment','Maïs','Betterave sucrière','Vulpin des champs','Moutarde des champs','Gaillet gratteron','Stellaire intermédiaire','Chénopode blanc','Agrostide des champs','Matricaire perforée','Capselle bourse-à-pasteur','Géranium fluet'],
                                'Culture/Adventice':['Culture','Culture','Culture','Adventice','Adventice','Adventice','Adventice','Adventice','Adventice','Adventice','Adventice','Adventice']})

    st.table(tableau_espece)

    st.markdown("<h1 style='font-size: 30px; color : green; text-decoration: underline;'> Objectifs :  </h1>", unsafe_allow_html=True)
    st.write("")

    st.write("""
    L’objectif de ce projet est donc de mettre au point un modèle capable d’attribuer à chaque photographie de semis l’espèce correspondante. Il s’agit donc d’un problème de **classification de données 
    non-structurées** (images), dont la variable cible est l’espèce végétale qui prend **12 modalités** (classification multi-classe).""")
    

    st.markdown("<h1 style='font-size: 30px; color : green; text-decoration: underline;'> Références :  </h1>", unsafe_allow_html=True)
    st.markdown("V2 Plant Seedlings Dataset. (2018, 13 décembre). Kaggle. https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset")

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

