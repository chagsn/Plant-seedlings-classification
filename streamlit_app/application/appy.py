import streamlit as st 
import matplotlib.pyplot as plt
from skimage.transform import resize



from appy_DATA import Data_Visualisation
from appy_intro import Intro
from appy_LDA import reduction_lda
from appy_segmentation import segmentation 
from appy_1er import Home
from appy_modelisation import modelisation
from appy_Prediction import prediction
from appy_Conclusion import conclusion
from appy_result_seg import prediction_segmentation
from appy_Unet import Unet


with open("style.css", "r") as f:
    style = f.read()

st.set_page_config("Reconnaissance De Plantes",page_icon=":seedling:")

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

st.sidebar.title('Projet Reconnaissance De Plantes')
st.sidebar.write("")


st.sidebar.image("image_premiere_page.png")

st.sidebar.write("")
st.sidebar.write("")

liste_menu =[ 'Home','Contexte', 'Data Visualisation', 'Reduction de dimension par LDA',"Modèles et Amélioration","Démonstration des modèles",
             "Segmentation par Deep Learning","Résultat de la segmentation","Conclusion","Annexe : la segmentation par seuillage"]

menu = st.sidebar.selectbox('__Menu :__ ', liste_menu)

if menu == liste_menu[0] : 
    Home()

if menu == liste_menu[1] : 
    Intro()

if menu == liste_menu[2] :
    Data_Visualisation()

if menu == liste_menu[3] :
    reduction_lda()

if menu == liste_menu[4] :
    modelisation()

if menu == liste_menu[5] :
    prediction()

if menu == liste_menu[6] :
    Unet()

if menu == liste_menu[7] :
    prediction_segmentation()

if menu == liste_menu[8] :
    conclusion()

if menu == liste_menu[9] :
    segmentation()




st.sidebar.write("")


authors = "__Auteurs__ :<br>Camille Lê<br>Charlotte Guesneau<br>Thi Thao Truc Le<br>Mariotte Zammit"
st.sidebar.write(authors, unsafe_allow_html=True)

tutor = "__Tuteur__ :<br>Maxime @Datascientest"
st.sidebar.write(tutor, unsafe_allow_html=True)

st.sidebar.write("")


img = plt.imread('logo.png')
new_size = (60,60)  
resized_img = resize(img, new_size)

st.sidebar.image(resized_img)
