def Data_Visualisation() :   
    import streamlit as st
    import pandas as pd
    import numpy as np
    from PIL import Image

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    
    import statsmodels
    import plotly.express as px

    ## Titre 
    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'> Data Visualisation </h1>", unsafe_allow_html=True)

    ## Introduction
    st.write ("Pour l'étape de visualisation, toutes les données créées lors du traitement des images ont été stockées dans un Dataframe afin de faciliter l'étape de visualisation. ")

    ## Présentation du DataFrame
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Présentation du Dataframe :  </h1>", unsafe_allow_html=True)
    st.write(' Le dataframe regroupant un total de 12 colonnes : ')

    df = pd.read_csv('V2_Plant_Seedlings_DataFrame.csv')
    st.dataframe(df.head(15))
    
    st.write("""•	_filepath_ :  permet d’accéder à chaque image du jeu de donnée par le biais de son chemin d’accès vers le fichier correspondant.""")
    st.write("""•	_species_ : cumule 12 modalités correspondant à l’espèce de la plante représentée sur l’image en question.""")
    st.write("""•	_image_name_ : est le nom de l'image.""")
    st.write("""•	_height_ et _width_ : sont respectivement la hauteur et la taille de l’image. """)
    st.write("""•	_image_size_ : est le nombre de pixels composant l’image.""")
    st.write("""•	_square_ : possède 2 modalités et décrit si l’image est carrée ou non.""")
    st.write("""•	_R_mean_, _V_mean_ et _B_mean_ : désignent la moyenne de l’intensité lumineuse des pixels de l’image dans le codage RVB, c’est-à-dire l’intensité d’un pixel de l’image respectivement en rouge, 
        vert et bleu.""")
    st.write("""•	_luminosity_ : quantifie la luminosité de l’image : plus la valeur est élevée, moins l’image  est sombre.""")
    st.write("""•	_sharpness_score_ quantifie la netteté d’une image : plus la valeur est élevée, plus l’image est nette.""")

    st.write ("")
    st.write ("")

    ## Inspection de la composition du dataset
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Inspection de la composition du dataset:  </h1>", unsafe_allow_html=True)
    st.write ("")

    # Count_plot pour avoir le nombre d'image par espéce 
    count_plot, ax_plot = plt.subplots()
    sns.set_style("whitegrid")
    ax_plot = sns.countplot(x='species', palette="summer", data=df, order=df.value_counts("species").index)
    plt.xticks(rotation=65)
    plt.title("Nombre d'images par espèce", fontsize = 10, fontweight = 'extra bold')
    st.pyplot(count_plot)

    st.write((df.species.value_counts(normalize=True).round(4))*100)
    st.caption("Pourcentage d'image par classe")
    
    with st.expander("Explication :"):

        st.write("""On constate une certaine disparité entre les classes d'espèce de plantes : si la moyenne d'images par dossier avoisine les 450, les 5539 images composant le dataset ne sont pas distribuées 
            équitablement. De fait, Loose Silky-bent qui est la classe la plus commune possède 3 fois plus d'images que Common wheat, la classe minoritaire de notre dataset. Les espèces issues de l'agriculture 
            (Wheat, Maize & Beet) sont en minorité comparées aux mauvaises herbes.""")

        st.write("""Le rapport 3 entre les effectifs de la classe majoritaire (Loose Silky-bent) et ceux de la classe minoritaire (Common wheat) nous conduit néanmoins à considérer que le déséquilibre observé 
            entre les classes demeure relativement modéré.""")

    st.write ("")
    st.write ("")

    ## Quelques images de la base de donnée
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Quelques images de la base de données :  </h1>", unsafe_allow_html=True)
    st.write ("")

    # Selecteur pour selectionner l'espèce dont on veut afficher les imagees 
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

    #Affichage des image en fonction de l'espece choisie
    def choix_image() : 
        indices = indice(4, df, len(df['species'].unique()))
        fig, ax = plt.subplots(nrows=int(len(indices)/4), ncols=4,figsize=(15, 40))

        for i, idx in enumerate(indices):
            filepath = df["filepath"][idx]
            img = plt.imread(filepath)
            ax[i//4, i%4].imshow(img)
            ax[i//4, i%4].set_title(df['species'][idx])
            ax[i//4, i%4].grid(False)

        st.pyplot(fig)

    if st.button('Reset',key="image base de données"):
        choix_image()
    else : 
        choix_image()


    with st.expander("Explication :"):
        st.write("""
        Plusieurs observations peuvent être émises :

        •	Les photos sont prises en vue du dessus.

        •	Les dimensions des images varient singulièrement.

        •	Certaines images sont plus floues que d'autres, le flou pouvant atteindre un niveau élevé pour certaines photographies.

        •	Il ne semble pas y avoir de fond propre à une classe : le fond de chaque image paraît composé de cailloux et/ou d'une barquette et ce, indépendamment de la variété qu'elle représente.

        •	On note également la présence ponctuelle d'une bande de mesure issue de la barquette, qui prend une grande partie sur certaines images.

        •	En comparant la taille des cailloux composant le fond de la plupart des images, on comprend que certaines photographies ont été prises plus proches du sol que d'autres.

        •	Les 12 variétés exhibent des plantes d'une teinte de vert similaire. De plus les cailloux sont aussi toujours dans les mêmes tons.

        •	Il arrive qu'une image dispose d'un fond où le noir et le blanc dominent, cela est dû au mesureur de la barquette ainsi que du sol.

        •	Les photographies ont été prises à différentes étapes du cycle de croissance des plantes. Par exemple, pour la même espèce, certaines images montrent une seule feuille tandis que d’autres 
            montrent un spécimen plus mature avec plusieurs grandes feuilles.""")

    st.write ("")
    st.write ("")

    ## Dimensions des images
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'>Dimensions des images : </h1>", unsafe_allow_html=True)
    st.write ("")

    with st.expander("Graphique :"):
        taille_image = px.scatter(df, x="width", y="height", color="species", size="width", hover_name="image_name")
        taille_image.update_layout(title=dict(text="Dimension des images par espèce", font=dict(size=20), x = 0.2))

        st.plotly_chart(taille_image)

        st.write("""La majorité des images ne font pas plus de __1500 par 1500 pixels__.
        La plus petite image fait __49x49 pixels__ tandis que la plus grande fait __3652x3457 pixels__. Nous avions précédemment émis le constat que les dimensions des images variaient sensiblement. Nous pouvons à présent également affirmer que la grande majorité des __images sont carrées__. Seules quelques-unes semblent ne pas l'être. Identifions-les.
        """)

    # Selecteur  pour la repartitions de la taille 
    classe_taille = np.append('Toutes les classes',df['species'].unique())
    option_taille = st.selectbox('Selectionner une espece  : ', classe_taille, key='espece_selectionnee')

    if option_taille == 'Toutes les classes':
        species = df['species'].unique()
        fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(20, 40))
        ax = ax.flatten()

        for i in range(len(species)):
            ax[i].scatter(x='width', y='height', data=df[(df['species'] == species[i]) & (df['width'] == df['height'])], color="blue")
            ax[i].scatter(x='width', y='height', data=df[(df['species'] == species[i]) & (df['width'] != df['height'])], color="red")
            ax[i].set_title('Taille des images pour les '+ species[i],fontsize=20)
            ax[i].set_xlabel('width')
            ax[i].set_ylabel('height')
            ax[i].legend(["square", "not square"], loc="best",fontsize=20)

    else:
        species = option_taille
        st.write(species)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        ax.scatter(x='width', y='height', data=df[(df['species'] == species) & (df['width'] == df['height'])], color="blue")
        ax.scatter(x='width', y='height', data=df[(df['species'] == species) & (df['width'] != df['height'])], color="red")
        ax.set_title('Taille des images pour les '+ species,fontsize=12)
        ax[i].set_xlabel('width')
        ax[i].set_ylabel('height')
        ax.legend(["square", "not square"], loc="best")

    st.pyplot(fig)

    st.write ("")

    with st.expander("Explication :"):
        st.write("""Sur ce graphique, les points bleus représentent les images carrées et les points rouges représentent les images non carrées. Chaque graphique représente une espèce. Il y a un total de __68 images non 
            carrées ce qui correspond à 0.01% des images du Dataset__. Les variétés Sugar beet, Loose Silky-bent ainsi que Black-grass cumulent quasiment à elles trois l'intégralité des images non carrées. 
            Black-grass, en particulier, frappe par son fort taux d'images 
            aux dimensions asymétriques, elle qui fait partie des classes minoritaires du dataset.
            Puisque la quasi-totalité des images sont carrées, nous nous baserons à présent arbitrairement sur la hauteur afin d'évaluer la distribution de la taille des images.""")

    st.write ("")
    st.write ("")

    # box plot pour la distribution de la taille des images
    with st.expander("box plot :"):
        fig = px.box(df, x='height', color='species', hover_name="image_name")
        fig.update_layout(title=dict(text="Distribution de la taille des images par espèce", font=dict(size=20), x=0.2))
        st.plotly_chart(fig)

        st.write ("")

        st.write ("""
        Le boxplot met en lumière :

        •	L'existence d'outliers pour chaque classe.

        •	Une distribution inconsistante entre les classes.

        •	Une taille minimum assez similaire pour toutes les classes (~ 60 pixels), à l'exception de Charlock.

        •	La distribution de la taille des images varie donc fortement d'une espèce à une autre. 

        •	Certaines telles que Common Chickweed ou Scentless Mayweed sont composées d'images de tailles semblables. 

        •	D'autres à l'inverse exibent des images de dimensions très variées comme Common Wheat, Maize ou encore Black-grass qui avait déjà été citée précédemment pour son taux anormalement 
        élévé d'images de dimensions asymétriques.
    """)

    st.write ("")
    st.write ("")

    ## Flou des images
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'>Flou des images:  </h1>", unsafe_allow_html=True)
    st.write ("")

    fig = px.scatter(df, x="sharpness_score", y="image_size", color="species", hover_name="image_name")
    fig.update_layout(title=dict(text="Relation entre les dimensions et la netteté de l'image par espèce", font=dict(size=20), x=0.1),
                    xaxis_title="Dimensions",
                    yaxis_title="Sharpness_score")
    st.plotly_chart(fig)

    with st.expander("Explication :"):
        st.write ("""
            On constate une relation de linéarité claire entre les 2 variables: de fait, plus une image est petite et plus elle est floue.
            
            __Une théorie serait que les plus petites images du dataset soient en réalité issues d'un tronquage manuel de photographies initialement plus grandes comportant plusieurs specimens, dans le but d'afficher des images de plantes individuelles__.
        """)

    st.write ("")
    st.write ("")

    ##Colorimétrie et Luminosité
    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'>Colorimétrie & Luminosité:  </h1>", unsafe_allow_html=True)
    st.write ("")

    # Barplot des moyennes des couleurs primaires 

    # Fonction pour calculer la moyenne de chaque couleurs primaire pour chaque espèces
    def calcul_moyenne (col_num,col_qual) : 
        espece = col_qual.unique()
        a = 0 
        moyenne ={}
        b = []
        for i, j in zip(col_num,col_qual) : 
            if j == espece[a] :
                b.append(i)
                moyenne[j] = sum(b)/len(b)
            else : 
                a = a + 1 
                b = []
        return moyenne

    # Calcul des moyennes de chaque couleur primaire (R,V,B) pour chaque espèce (classe)
    R = calcul_moyenne(df['R_mean'], df['species'])
    V = calcul_moyenne(df['G_mean'], df['species'])
    B = calcul_moyenne(df['B_mean'], df['species'])


    # Création des données pour le barplot
    species = df['species'].unique()
    values_R = R.values()
    values_V= V.values()
    values_B = B.values()

    # Affichage d'un barplot des moyennes de chaque couleur (R,V,B) pour chaque espèce
    fig_bar,ax_bar = plt.subplots()
    ax_bar.bar(species, values_R,label='moy_Rouge', color = 'red')
    ax_bar.bar(species,values_V, label='moy_Vert', color ='green')
    ax_bar.bar(species, values_B, label='moy_Bleu', color = 'blue')

    plt.legend(fontsize = 'x-small', loc = 'lower left')
    plt.xticks(species)
    plt.xlabel('Species')
    plt.ylabel('Moyenne des couleurs')
    plt.xticks(rotation=90)
    plt.title("Moyenne des couleurs primaires pour chaque espèce", fontsize = 10, fontweight = 'extra bold')

    st.pyplot(fig_bar)

    with st.expander("Explication :"):
        st.write ("""La colorimétrie des images ne varie pas sensiblement d'une classe à une autre. Cela confirme les observations antérieures.__Toutes les plantes sont de la même teinte de vert.__ 
        Il est également impossible de distinguer une classe d'une autre par le biais des couleurs du fond car celui-ci est toujours composé soit de cailloux de mêmes teintes, soit de noir, et ce, de manière tout à fait aléatoire. __Ainsi il n'est pas envisageable de se baser sur la colorimétrie des images afin de les classifier.__
        """)


    st.write ("")
    st.write ("")

    #Graphique de densité sur la distribution de la luminosité 
    fig_kde,ax_kde = plt.subplots()
    ax_kde = sns.kdeplot(x ='luminosity', data = df, hue = 'species')
    plt.title('Distribution de la luminosité pour les différentes espèces')
    st.pyplot(fig_kde)

    with st.expander("Explication :"):

        st.write (""" 
        La luminosité d'une image du dataset oscille en moyenne entre 60 et 80 indépendamment de la variété à laquelle elle appartient. La variété Cleavers présente la distribution de luminosité la plus étroite (images de luminosité proches). Les variétés suivantes présentent les répartitions de luminosité les plus étendues: Fat Hen, Common wheat, Black-grass. 
        __Sur l’ensemble du dataset, les images présentent pour la majorité une luminosité similaire.__
        """)

    st.write ("")
    st.write ("")

    with st.expander("Conclusion :"):
        ## Conclusion 
        st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'>Synthèse et perspectives :  </h1>", unsafe_allow_html=True)
        
        st.write ("""
        Notre jeu de données étant constitué de données non-structurées (images), le preprocessing sera concrètement effectué directement au moment de la modélisation, en entrée des algorithmes.
        L'exploration des données a cependant permis de mettre en lumière certaines caractéristiques importantes du jeu de données, qui nous permettront d'orienter nos choix de preprocessing et de modélisation, 
        et pourront également nous guider pour l'interprétation des résultats: 

        Les 12 classes de la variable cible ne sont pas parfaitement équilibrées, mais le niveau de déséquilibre reste modéré et ne semble a priori pas nécessiter de traitement spécifique à l'étape de modélisation. 
        Ce point restera à confirmer à partir des premiers résultats.

        Les images comportent toutes un arrière-plan similaire, constitué de cailloux et éventuellement d'une barquette, parfois accompagnée d'une bande de mesure. On peut se poser la question d'une éventuelle 
        suppression de ce fond avant classification des images. Ce point est développé un peu plus bas.

        La très grande majorité des images sont carrées, nous considérerons donc le format carré comme le format standard de notre dataset. Du fait de l'existence d'une relation statistiquement significative entre 
        la dimension des images et la variable cible (espèce d'appartenance), nous procéderons à un redimensionnement systématique des images au moment du preprocessing. Ainsi, le modèle pourra se lancer avec les images 
        redimensionnées.

        Certaines images présentent un niveau de flou élevé, qui pourrait éventuellement dégrader les performances du modèle. Ce point est à garder en tête au moment de l'analyse des résultats: les images les moins 
        bien classées correspondent-elles aux images floues ?

        L'étude du flou a également permis d'émettre l'hypothèse que les plus petites images sont en fait des images zoomées a posteriori par troncage. Dans ce cas le niveau de zoom (et donc la taille des cailloux qui
        constituent le fond des images) serait corrélé à la dimension des images, elle-même fortement corrélée à l'espèce d'appartenance. Ce point est également à garder en tête au moment de l'analyse des résultats: 
        il sera intéressant de vérifier que le modèle ne se base pas sur le fond des images (par exemple sur la taille des cailloux) pour effectuer la classification. Dans le cas contraire il faudra envisager une 
        étape supplémentaire de suppression de l'arrière-plan dans le preprocessing.

        Enfin les différentes classes d'images présentent des niveaux de luminosité très similaires, il ne sera donc a priori pas nécessaire de corriger la luminosité au moment du preprocessing. La distribution des 
        couleurs est également bien équilibrée entre les classes et ne constitue donc pas un critère discrimimant pour la classification.""")

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")