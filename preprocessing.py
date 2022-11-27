"""
En este archivo creamos la version 3 del dataset. En esta versión, se aplican los siguientes cambios:

Se ha detectado un nombre de la canción en inglés en el 51.4% de las canciones. El resto de valores se encuentran muy
repartidos. Por tanto, de cara a la predicción language se codificará como binaria (1 si se ha detectado inglés,
0 si no).

La columna `num_syllables` presenta un 12.1% de valores faltantes. Esto ha ocurrido en títulos de canciones que
estaban en idiomas no soportados por la librería que contaba las sílabas. Estos valores se sustituirán por la media y
se añadirá una columna bandera que indique si el valor era faltante.

En cuanto al análisis de sentimiento, hemos utilizado una librería textblob que asigna  un valor entre -1 y 1
en función de las palabras que encuentra (polaridad). Para ello, asigna a cada palabra un valor, el resultado total
es la suma de todas ellas. La mayoría de canciones, un 78.2% han sido etiquetadas con una polaridad de 0. Es por ello
que vamos a redondear todas las palabras con un sentimiento total negativo a -1 y viceversa.

Aplicaremos una transformación Box Cox a las variables: acousticness, instrumentalness, liveness, speechiness y
loudness. Esto hará que sigan distribuciones más gaussianas ya que presentan distribuciones algo asimétricas.
Esto es útil cuando realicemos regresión lineal puesto que una de las suposiciones que el modelo hace sobre las
variables es que siguen una distribución gaussiana.

La columna time_signature, puesto que representa el compás de la canción, hemos decidido codificarla como una columna
binaria que indique si es 4/4 o no, debido a que el 94% de ellas lo son. La columna key, en cambio, sí que hemos decidido
mantenerla así, puesto que entendemos que puede haber más similitud entre canciones que usan notas parecidas,
y además porque realizar un one-hot crearía 12 características nuevas. Esto se debe a que, al seguir una distribución
relativamente uniforme, no es posible crear una característica donde agrupar a las infrecuentes.

Una vez hechas estas transformaciones aplicaremos un MinMaxScaler a todas las columnas cuyo rango no esté en el rango
[0, 1] o en valores negativos centrados en cero. Es decir, lo aplicaremos a: key, tempo, num_syllables,
avg_chars_per_word y num_uppercase_words.

La columna num_lowercase_words, la estandarizaremos para conservar los valores cero cercanos a cero.

En song_duration_ms, observamos que hay canciones que son mucho más largas que otras por lo que será útil poner
un tope a estos valores atípicos. Esto se limitará a 3 desviaciones típicas con respecto a la media, ya que la
distribución que sigue es gaussiana. Tras esto, se aplicará un MinMaxScaler (queremos seguir manteniendo los valores
cero a cero).

Este preprocesamiento se hará haciendo un fit a los estimadores en el conjunto de entrenamiento y transformando
ambos conjuntos. Esto se hace para evitar transformar el conjunto de entrenamiento con información del conjunto de
test.

En avg_chars_per_word, cuando no hay palabras, hay valores faltantes, estos valores serán rellenados con cero.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from pandas_profiling import ProfileReport

from utils import load_data


def preprocess_phase1(df, minimalist=False):
    # Eliminamos song_name
    df = df.drop(columns=['song_name'])

    # Codificamos la columna language
    df['language'] = (df['language'] == 'en').astype(int)

    # Imputamos los valores faltantes de num_syllables
    df['num_syllables_missing'] = df['num_syllables'].isnull().astype(int)
    df['num_syllables'] = df['num_syllables'].fillna(df['num_syllables'].mean())

    if not minimalist:
        # Redondeamos la polaridad a -1 o 1, si es cero se deja igual
        neg = df['sentiment'] < 0
        pos = df['sentiment'] > 0
        df.loc[neg, 'sentiment'] = -1
        df.loc[pos, 'sentiment'] = 1

    # Codificamos la columna time_signature
    df['time_signature'] = (df['time_signature'] == 4).astype(int)

    # Rellenamos los valores faltantes de average_chars_per_word con cero
    df['avg_chars_per_word'] = df['avg_chars_per_word'].fillna(0)

    return df


def preprocess(train, test):
    train = preprocess_phase1(train)
    test = preprocess_phase1(test)

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
