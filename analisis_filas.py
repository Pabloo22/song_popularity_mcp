"""
En este script se comprueba si hay filas duplicadas en el dataset. Para ello, se utiliza la
función duplicated() de pandas.
"""

import pandas as pd

from utils import load_data


def main():
    # Cargamos los datos
    train, _= load_data(exclude_id=True, split=False)

    # Comprobamos si hay filas duplicadas excluyendo el id de la canción 'song_id'
    print('Número de filas duplicadas: ', train.duplicated().sum())

    # Eliminamos las filas duplicadas
    train = train.drop_duplicates()

    # Guardamos el dataset sin filas duplicadas
    train.to_csv('data/train_no_duplicates.csv', index=False)


if __name__ == '__main__':
    main()
