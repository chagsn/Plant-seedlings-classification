
def conclusion() :

    import streamlit as st 

    st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;text-decoration: underline;'> Conclusion  </h1>", unsafe_allow_html=True)



    ## Les difficultés 

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Les difficultés rencontrées :  </h1>", unsafe_allow_html=True) 

    st.markdown("- L'implémentation du Grad-CAM pour l'interprétabilité")
    st.markdown("- L'augmentation des données")
    st.markdown("- La segmentation")




    ## Les limites des modéles  

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Les limites des modèles :  </h1>", unsafe_allow_html=True) 

    st.markdown("- La qualité des images")
    st.markdown("- La maturité des plantes")
    st.markdown("- L'angle de prise de vue")
    st.markdown("- Les variétés inter espèces")
    st.write("=> La base de données est trop limitée en termes d'informations.")


    ## Les perspectives   

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Les perspectives :  </h1>", unsafe_allow_html=True) 

    st.markdown("- La mise en place de la méthode SHAP (SHapley Additive exPlanations)")
    st.markdown("- Agrandir la base de données")
    st.markdown("- Améliorer l'entraînement Unet avec une base de données hétérogène")
    st.markdown("- Entrainer les modèles avec de la segmentation par Deep Learning")



    ## Pour aller plus loins   

    st.markdown("<h1 style='font-size: 25px; color : green; text-decoration: underline;'> Pour aller plus loin :  </h1>", unsafe_allow_html=True) 

    st.markdown("- Intégrer d'autres espèces à la base de données")
    st.markdown("- Reconnaissance de maladies")
    st.markdown("- Identifier la maturité des plantes")

    st.write("")
    st.write("")
    st.write("")


    if st.button('Fin'):
        st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;'> Merci pour votre attention   </h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size: 40px; color : blue;'> Des Questions ?  </h1>", unsafe_allow_html=True)

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
