import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from pandas_profiling import ProfileReport

from utils import load_data


def preprocess_phase1(df, minimalist=False, updated=False):
    # Eliminamos song_name
    df = df.drop(columns=['song_name'])

    # Codificamos la columna language
    if not updated:
        df['language'] = (df['language'] == 'en').astype(int)

    # Imputamos los valores faltantes de num_syllables
    df['num_syllables_missing'] = df['num_syllables'].isnull().astype(int)
    df['num_syllables'] = df['num_syllables'].fillna(df['num_syllables'].mean())

    if not minimalist and not updated:
        # Redondeamos la polaridad a -1 o 1, si es cero se deja igual
        neg = df['sentiment'] < 0
        pos = df['sentiment'] > 0
        df.loc[neg, 'sentiment'] = -1
        df.loc[pos, 'sentiment'] = 1

    # Codificamos la columna time_signature
    if not updated:
        df['time_signature'] = (df['time_signature'] == 4).astype(int)

    # Rellenamos los valores faltantes de average_chars_per_word con cero
    df['avg_chars_per_word'] = df['avg_chars_per_word'].fillna(0)

    return df


def preprocess(train, test, updated=False):
    # Nota: la versión updated no ha sido utilizada finalmente
    train = preprocess_phase1(train, updated=updated)
    test = preprocess_phase1(test, updated=updated)

    # Aplicamos una transformación Box Cox a las siguientes columnas
    cols = ['acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']
    pt = PowerTransformer()
    train[cols] = pt.fit_transform(train[cols])
    test[cols] = pt.transform(test[cols])

    # Estandarizamos la columna num_lowercase_words
    std_scaler = StandardScaler()
    train['num_lowercase_words'] = std_scaler.fit_transform(train['num_lowercase_words'].values.reshape(-1, 1))
    test['num_lowercase_words'] = std_scaler.transform(test['num_lowercase_words'].values.reshape(-1, 1))

    # Aplicamos un tope a song_duration_ms y a num_syllables
    if not updated:
        mean_duration = train['song_duration_ms'].mean()
        std_duration = train['song_duration_ms'].std()
        train['song_duration_ms'] = np.clip(train['song_duration_ms'], 0, mean_duration + 3 * std_duration)
        test['song_duration_ms'] = np.clip(test['song_duration_ms'], 0, mean_duration + 3 * std_duration)

        mean_syllables = train['num_syllables'].mean()
        std_syllables = train['num_syllables'].std()

        train['num_syllables'] = np.clip(train['num_syllables'], 0, mean_syllables + 3 * std_syllables)
        test['num_syllables'] = np.clip(test['num_syllables'], 0, mean_syllables + 3 * std_syllables)

    # Aplicamos un MinMaxScaler a song_duration_ms
    min_max_scaler = MinMaxScaler()
    train['song_duration_ms'] = min_max_scaler.fit_transform(train['song_duration_ms'].values.reshape(-1, 1))
    test['song_duration_ms'] = min_max_scaler.transform(test['song_duration_ms'].values.reshape(-1, 1))

    # Aplicamos un MinMaxScaler a las columnas que no estén en el rango [0, 1]
    cols = ['key', 'tempo', 'num_syllables', 'avg_chars_per_word', 'num_uppercase_words']
    min_max_scaler = MinMaxScaler()
    train[cols] = min_max_scaler.fit_transform(train[cols])
    test[cols] = min_max_scaler.transform(test[cols])

    if updated:
        # Codificamos la columna 'language' con la frecuencia de aparición de cada idioma
        train['language'] = train['language'].map(train['language'].value_counts(normalize=True))
        test['language'] = test['language'].map(train['language'].value_counts(normalize=True))

    return train, test


def main():

    train, X_test = load_data(split=False, version=2)

    new_train, new_test = preprocess(train, X_test)

    new_train.to_csv('data/train_preprocessed.csv', index=False)
    new_test.to_csv('data/test_preprocessed.csv', index=False)

    # Creamos un report nuevo para comprobar que los cambios se han realizado correctamente
    profile = ProfileReport(new_train,
                            title='train_preprocessed report',
                            html={'style': {'full_width': True}},
                            minimal=True)
    profile.to_file('reports/train_preprocessed_report.html')


def preprocessing_v2():
    """
    Este preprocesado está pensado para algoritmos que no realizan ninguna suposición sobre las distribuciones de
    las variables independientes. La idea es realizar un preprocesado que simplemente añada las características
    extraídas a partir del nombre de la canción y rellene los valores nulos generados.
    """

    train, X_test = load_data(split=False, version=2)
    train = preprocess_phase1(train, minimalist=True)
    X_test = preprocess_phase1(X_test, minimalist=True)

    # Guardamos los datos preprocesados
    train.to_csv('data/train_preprocessed_v2.csv', index=False)
    X_test.to_csv('data/test_preprocessed_v2.csv', index=False)

    # Creamos un report nuevo para comprobar que los cambios se han realizado correctamente
    profile = ProfileReport(train,
                            title='train_preprocessed_v2 report',
                            html={'style': {'full_width': True}},
                            minimal=True)
    profile.to_file('reports/train_preprocessed_v2_report.html')


if __name__ == '__main__':
    preprocessing_v2()
