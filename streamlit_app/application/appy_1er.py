
def Home() : 
    import streamlit as st
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import numpy as np 


    st.markdown("<h1 style='text-align: center; font-size: 48px; color : green;text-decoration: underline;'>RECONNAISSANCE DES PLANTES</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 32px; color : red;'>Projet fil rouge </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 16px; color : black;'>Mars 2023 </h1>", unsafe_allow_html=True)

    st.write("")



    df = pd.read_csv("V2_Plant_Seedlings_DataFrame.csv")

    def indice(x, df, num_of_species=12):
    # Fonction retournant x indices d'images aléatoires par espèce d'un DataFrame
        if num_of_species > 12:
            print("Il n'y a que 12 espèces.")

        elif x > min(df.species.value_counts()):
            print("Le dataset ne contient que ",df.species.value_counts().sort_values()[0], df.species.value_counts().sort_values().index[0], ".")
    
        else:
            indices = []
            for s in np.random.choice(df.species.unique(), size = num_of_species, replace=False):
                    indices += list(np.random.choice(df[df.species == s].index, size = x, replace=False))
            return indices 

    indices = indice(1,df)  
    fig, ax = plt.subplots(nrows= 3, ncols= 4,figsize=(20, 60))
    fig.subplots_adjust(hspace=-0.9, wspace=0.2)
    for i, idx in enumerate(indices):
        filepath = df["filepath"][idx]
        img = plt.imread(filepath)
        ax[i // 4, i % 4].imshow(img)  
        ax[i // 4, i % 4].set_title(df['species'][idx],fontsize=16, fontweight='bold')
        ax[i // 4, i % 4].grid(False)
        ax[i // 4, i % 4].set_xticks([])
        ax[i // 4, i % 4].set_yticks([])
        
    st.pyplot(fig)
    



