"""
Además de las características que hemos añadido anteriormente, vamos a añadir nuevas características
que son la combinación de las características que ya tenemos. Esta combinación se realizará mediante
el producto cartesiano de las características. Por ejemplo, si tenemos las características 'a' y 'b',
la combinación de estas características será 'a * b'. Esto se hará para las variables que tienen que ver
con propiedades sonoras de la canción que sean continuas y estén en un rango [0, 1]. Por ejemplo,
la variable 'acousticness' y 'danceability'.
"""
from itertools import product

from utils import load_data


def main():

    selected_cols = ['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'valence']