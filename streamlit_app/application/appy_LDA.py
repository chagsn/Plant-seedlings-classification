

def reduction_lda() : 
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

    ## Titre 
    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'> Réduction de dimension par LDA </h1>", unsafe_allow_html=True)

    st.write("""
        L’appplication d’un algorithme de LDA ou analyse discriminante linéaire permet de projeter les données de telle sorte que la variance au sein d’une classe soit la plus faible 
        possible, tout en maximisant la variance entre les classes.""")


    ## ouverture du DataFrame
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Les données :  </h1>", unsafe_allow_html=True)
    st.write(' Le dataframe regroupant un total de 12 colonnes.')

    df = pd.read_csv('V2_Plant_Seedlings_DataFrame.csv')
    st.dataframe(df.head())


    class2label = {x : y for y, x in enumerate(df.species.unique())}


    st.write("""
    Le dataframe comporte 5539 liens qui retournent des images en couleurs.""")

    st.write("""
    Pour cette étape on a appliqué la LDA sur :
        
    •   Les images brutes redimensionnées en 100\*100\*3
        
    •   Les images segmentées et redimensionnées en 100\*100\*3
        
    Les résultats de la LDA ont été stockés dans des dataframes.""")

    st.write(" ")
    st.write(" ")

    ## La reduction sans segmentation
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> LDA sans segmentation :   </h1>", unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")

    # dataframe après réduction par LDA
    df_lda = pd.read_csv("data_lda.csv")
    st.dataframe(df_lda.head())
    st.write("""__Après la réduction de dimension chaque image est exprimée par__ """,df_lda.shape[1],"__variables__")

    # Format des donnnée pour représentation graphique
    df_lda = np.loadtxt('data_lda.csv', delimiter=',')
    target = np.array([class2label[i] for i in df.species.values])

    st.write(" ")

    # graphique 2d
    fig_2d, ax_2d = plt.subplots()
    ax_2d = plt.scatter(df_lda[:, 0], df_lda[:, 1],  c = target, cmap='Set2')
    plt.xlabel('LDA 1')
    plt.ylabel('LDA 2')
    plt.title("Données projetées sur les 2 axes de LDA")
    cbar = plt.colorbar(ax_2d, ticks=range(12))
    cbar.ax.set_yticklabels(df.species.unique())
    st.pyplot(fig_2d) 

    st.write(" ") 

    # graphique 3d
    fig = px.scatter_3d(df_lda, x=df_lda[:, 0], y=df_lda[:, 1], z=df_lda[:, 2], color=df.species, hover_name = df.image_name)
    fig.update_layout(title=dict(text="Projection 3D des données issues de la LDA", font=dict(size=25), x=0.2))
    st.plotly_chart(fig)

    with st.expander("Explication :"):
        st.write("""
            Ainsi, il est possible de se faire une représentation visuelle du dataset en projetant les données sur les 3 premières variables obtenues après réduction.
            L’interprétation graphique permet par exemple de voir qu’une Small-flowered Cranebill est très différente d’une Common wheat.
            À l’inverse les variétés Fat Hen et Sugar Beet semblent très similaires et pourraient être plus facilement confondues par un modèle de classification.
        """)

    st.write(" ")
    st.write(" ")


    ## La reduction après segmentation
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> LDA avec segmentation :   </h1>", unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")

    # dataframe après réduction par LDA
    df_lda_segmented = pd.read_csv("data_lda_segmented.csv")
    st.dataframe(df_lda_segmented.head())
    st.write("""Après la réduction de dimension chaque image est exprimée par """, df_lda_segmented.shape[1]," variables")

    # Format des donnnées pour représentation graphique
    df_lda_segmented = np.loadtxt('data_lda_segmented.csv', delimiter=',')
    target = np.array([class2label[i] for i in df.species.values])

    st.write(" ")

    # graphique 2d
    fig_2d, ax_2d = plt.subplots()
    ax_2d = plt.scatter(df_lda_segmented[:, 0], df_lda_segmented[:, 1],  c = target, cmap='Set2')
    plt.xlabel('LDA 1')
    plt.ylabel('LDA 2')
    plt.title("Données projetées sur les 2 axes de LDA")
    cbar = plt.colorbar(ax_2d, ticks=range(12))
    cbar.ax.set_yticklabels(df.species.unique())
    st.pyplot(fig_2d) 

    st.write(" ") 

    # graphique 3d
    fig = px.scatter_3d(df_lda, x=df_lda_segmented[:, 0], y=df_lda_segmented[:, 1], z=df_lda_segmented[:, 2], color=df.species, hover_name = df.image_name)
    fig.update_layout(title=dict(text="Projection 3D des données issue de la LDA", font=dict(size=25), x=0.2))
    st.plotly_chart(fig)

    with st.expander("Explication :"):
        st.write("""
        En comparant avec les résultats de la LDA obtenus sur le jeu de données original, on constate que le fait de ne conserver que la plante sur l’image en s’affranchissant du fond, jugé polluant, permet d’accentuer 
        drastiquement la maximisation -respectivement minimisation- de la variance interclasse -respectivement intraclasse-.
        De fait, tandis que nous obtenions auparavant des clusters de points, cette même représentation sur le jeu de données segmentées permet de voir que les classes se présentent dès lors quasiment sous forme de
        points dans l’espace tridimensionnel.
        Ce constat est prometteur quant aux futurs résultats de classification des modèles de Deep Learning avec ce nouveau pre-processing du jeu de données.
    """)
    
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
