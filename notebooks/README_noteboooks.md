# Plant seedlings classification project


## Presentation

This repository contains the code for our project **PLANT SEEDLINGS CLASSIFICATION**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

This project lies within the framework of plant recognition from pictures: the goal is to develop a **Computer Vision model** able to **classify a plant specie from a specimen photograph**.

It was developed by the following team :  
- Charlotte GUESNEAU ([GitHub](https://github.com/chagsn/) / [LinkedIn](www.linkedin.com/in/cguesneau/))
- Mariotte ZAMMIT ([LinkedIn](https://www.linkedin.com/in/mariotte-zammit/))
- Camille LE
- Thi Tha LE ([LinkedIn](https://www.linkedin.com/in/thi-tha-le-b20b84170/))


## Dataset
The dataset at the basis of the project is the Kaggle "V2 Plant Seedlings Dataset" avalaible at:  
https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset  

The previous (V1) version of this dataset was used in the Plant Seedling Classification playground competition on Kaggle.  

Complementary information about the dataset  can be found in the [data](./data) folder.


## Code
You can browse and run the [notebooks](./notebooks).  
To that end, you will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

The way the different notebooks are organized is described in the [README_notebooks.md](./notebooks/README_notebooks.md)  file.


## Streamlit App
We developped a dedicated web app using Streamlit framework to present our work during our Data Scientist certification oral defense.  
The corresponding code is available in the [streamlit_app](./streamlit_app) folder.  
To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```


## Documentation
All the work performed to carry out this project is fully documented in our Data Scientist certification final report, which can be shared on demand.
