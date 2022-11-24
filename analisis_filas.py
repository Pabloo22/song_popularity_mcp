"""
En este script se comprueba si hay filas duplicadas en el dataset. Para ello, se utiliza la
función duplicated() de pandas.
"""

import pandas as pd

from utils import load_data


def main():
    # Cargamos los datos
    X_train, _, _ = load_data(exclude_id=True)

    # Comprobamos si hay filas duplicadas excluyendo el id de la canción 'song_id'
    print('Número de filas duplicadas: ', X_train.duplicated().sum())

    # Eliminamos las filas duplicadas
    X_train = X_train.drop_duplicates()

    # Guardamos el dataset sin filas duplicadas
    X_train.to_csv('data/train_no_duplicates.csv', index=False)


if __name__ == '__main__':
    main()
