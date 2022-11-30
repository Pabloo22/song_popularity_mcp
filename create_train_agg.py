import pandas as pd
from pandas_profiling import ProfileReport

from utils import load_data


def create_train_agg():
    train, _ = load_data(exclude_id=True, split=False)

    # Agrupamos por 'song_name' y agregamos por la media en todas las columnas menos en 'key
    train_agg: pd.DataFrame = train.groupby('song_name').mean()
    # Guardamos el dataset
    train_agg.to_csv('data/train_agg.csv', index=True)
    train_agg.reset_index(inplace=True)

    # Generamos el reporte de pandas profiling
    profile = ProfileReport(train_agg,
                            title='train_agg Profiling Report',
                            html={'style': {'full_width': True}},
                            minimal=True)
    profile.to_file(output_file="reports/train_agg_report.html")


def create_train_agg2():
    train_agg, X_test = load_data(exclude_id=True, split=False, agg=1)

    # Eliminamos las filas cuyo nombre de la canción aparece en el dataset de testeo
    train_agg = train_agg[~train_agg['song_name'].isin(X_test['song_name'])]

    # Guardamos el dataset
    train_agg.to_csv('data/train_agg2.csv', index=False)

    # Generamos el reporte de pandas profiling
    profile = ProfileReport(train_agg,
                            title='train_agg2 Profiling Report',
                            html={'style': {'full_width': True}},
                            minimal=True)
    profile.to_file(output_file="reports/train_agg2_report.html")


def create_memorize_dataset():
    train, X_test = load_data(exclude_id=True, split=False)

    train = train[['song_name', 'song_popularity']]
    # Agrupamos por 'song_name' y agregamos por la media
    train_agg: pd.DataFrame = train.groupby('song_name').mean()
    train_agg.reset_index(inplace=True)

    # Guardamos el dataset
    train_agg.to_csv('data/memorize_dataset.csv', index=False)


if __name__ == '__main__':
    create_memorize_dataset()
