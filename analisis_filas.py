"""
En este script se comprueba si hay filas duplicadas en el dataset. Para ello, se utiliza la
función duplicated() de pandas.
"""

import pandas as pd

from utils import load_data


def main():
    # Cargamos los datos
    train, _ = load_data(exclude_id=True, split=False)

    # Comprobamos si hay filas duplicadas excluyendo el id de la canción 'song_id'
    print('Número de filas duplicadas: ', train.duplicated().sum())

    # Eliminamos las filas duplicadas
    train = train.drop_duplicates()

    # Guardamos el dataset sin filas duplicadas
    train.to_csv('data/train_no_duplicates.csv', index=False)


def print_top_songs():
    train, _ = load_data(exclude_id=True, split=False)

    # Imprimimos las 10 canciones más populares
    top = train[['song_name', 'song_popularity']].sort_values(by='song_popularity', ascending=False).drop_duplicates()
    print(top.head(10))

    # Imprimimos las 10 canciones menos populares
    print(top.tail(10))


if __name__ == '__main__':
    print_top_songs()
