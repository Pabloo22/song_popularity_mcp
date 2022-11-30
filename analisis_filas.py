"""
En este script se comprueba si hay filas duplicadas en el dataset. Para ello, se utiliza la
función duplicated() de pandas.
"""
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


def analysis_train_test():
    train, X_test = load_data(exclude_id=True, split=False, version=1, no_duplicates=True)
    X_train = train.drop(columns=['song_popularity'])
    print(X_test.columns == X_train.columns)
    X_test = X_test.drop(columns=['song_id'], errors='ignore')

    print("numero de filas duplicadas en test", X_test.duplicated().sum())
    print("song_popularity medio de las filas duplicadas en train", train[train.duplicated()]['song_popularity'].mean())
    # print("song_popularity medio de las filas duplicadas en test", X_train[X_train.duplicated()].song_popularity.mean())

    # Comprobamos si hay filas idénticas en train y test
    print('Número de filas idénticas en train y test:', X_train[X_train.isin(X_test)].dropna().shape[0])
    # Número de veces que coincide el nombre de la canción
    print('Número de veces que coincide el nombre de la canción:', X_train[X_train.song_name.isin(X_test.song_name)].shape[0])

    train_sample = X_train[X_train.song_name.isin(X_test.song_name)].sample(50, random_state=22)
    test_sample = X_test[X_test.song_name.isin(train_sample.song_name)]

    # Veamos el ejemplo de la canción 'Seven Nation Army'
    seven_nation_army = train_sample[train_sample.song_name == 'Seven Nation Army'].values
    print(seven_nation_army)
    print(test_sample[test_sample.song_name == 'Seven Nation Army'].values)


if __name__ == '__main__':
    analysis_train_test()
