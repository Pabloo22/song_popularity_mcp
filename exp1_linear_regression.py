"""
Uno de los objetivos tras este proyecto era obtener un mejor entendimiento de la regresión lineal. Es por ello que nos
hemos propuesto como un reto el tratar de obtener unos resultados competitivos utilizando únicamente este modelo.
No obstante, por los resultados obtenidos en el modelo base y el análisis realizado, hemos podido ver que la relación
entre las variables dependientes y la popularidad de la canción no es lineal. Es por esto, que necesitamos transformar
las variables de entrada de alguna forma. Es por eso que probaremos a crear nuevas variables que sean producto de las
características que tienen que ver con el nombre de la canción.
Una vez hecho esto, el pipeline se compondrá de los siguientes pasos:
Kernel PCA. Probaremos con todos los posibles núcleos (incluido el lineal) y nos quedaremos con el número de componentes
que tenga una varianza explicada de al menos el 95%.
Seleccionaremos las K mejores características.
Realizamos la predicción usando Regresión Lineal, Lasso, Ridge o Elastic Net. El motivo por el cuál no usamos
únicamente Elastic Net, es porque al elegir valores de regularización a cero se producen inestabilidades numéricas.
La optimización del kernel, del K y de los hiperparámetros de regularización de los respectivos modelos se hará
mediante una búsqueda en grid. Además, probaremos a utilizar las combinaciones o a no utilizarlas.
"""

import pandas as pd
from sklearn import (decomposition,
                     pipeline,
                     preprocessing,
                     model_selection,
                     feature_selection,
                     metrics,
                     linear_model)
from itertools import product
import numpy as np

from utils import load_data, save_prediction, print_grid_results


def feature_engineering(X_train, X_test):

    cols_to_combine = ['song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                       'loudness', 'speechiness', 'tempo', 'audio_valence']

    # Creamos nuevas columnas combinando las anteriores
    new_train_cols = [pd.DataFrame(X_train[col1] * X_train[col2], columns=[col1 + '_' + col2])
                      for col1, col2 in product(cols_to_combine, repeat=2) if col1 != col2]
    new_test_cols = [pd.DataFrame(X_test[col1] * X_test[col2], columns=[col1 + '_' + col2])
                     for col1, col2 in product(cols_to_combine, repeat=2) if col1 != col2]

    # Añadimos las nuevas columnas al dataset
    X_train = pd.concat([X_train] + new_train_cols, axis=1)
    X_test = pd.concat([X_test] + new_test_cols, axis=1)

    # Eliminamos las columnas originales
    X_train = X_train.drop(cols_to_combine, axis=1)
    X_test = X_test.drop(cols_to_combine, axis=1)

    return X_train, X_test


def main(feature_eng: bool = False):

    np.random.seed(2)

    X_train, y_train, X_test = load_data(split=True, preprocessed=True, exclude_id=True)

    if feature_eng:
        X_train, X_test = feature_engineering(X_train, X_test)
        print('Número de columnas: ', X_train.shape[1])

    # Creamos el pipeline:
    #   - PCA
    #   - Standard Scaler
    #   - Select K best
    #   - Elastic Net
    # Búsqueda de hiperparámetros:
    #   - PCA n_components: [0.95, 0.975, 0.99]
    #   - Select K best k: range(1, 15)
    #   - Select K bets score_func: [f_regression, mutual_info_regression]
    #   - Elastic Net alpha: [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1, 10, 100]
    #   - Elastic Net l1_ratio: [0.0001, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    # Creamos el pipeline
    pipe = pipeline.Pipeline([
        ('pca', decomposition.PCA()),
        ('select_k_best', feature_selection.SelectKBest()),
        ('elastic_net', linear_model.LinearRegression())])

    param_grid = {
        'pca__n_components': [0.95, 0.975, 0.99],
        'select_k_best__k': range(5, 20, 2),
        'select_k_best__score_func': [feature_selection.f_regression, feature_selection.mutual_info_regression],
    }
    # param_grid = {
    #     'k_pca__n_components': [0.95],
    #     'select_k_best__k': [10],
    #     'select_k_best__score_func': [feature_selection.f_regression],
    #     'elastic_net__alpha': [0.0001],
    #     'elastic_net__l1_ratio': [0.0001]
    # }

    rep_k_fold = model_selection.KFold(n_splits=5, shuffle=True)

    grid = model_selection.GridSearchCV(pipe, param_grid, scoring='r2', cv=rep_k_fold, n_jobs=-1, verbose=1)

    grid.fit(X_train, y_train)

    print_grid_results(grid)

    model = grid.best_estimator_
    model.fit(X_train, y_train)

    # Imprimimos el r2 score en train
    print('r2 score en train: ', model.score(X_train, y_train))
    print('mae en train: ', metrics.mean_absolute_error(y_train, model.predict(X_train)))

    y_pred = model.predict(X_test)

    # Guardamos las predicciones
    filename = 'exp1_linear_regression_no_comb.csv' if not feature_eng else 'exp1_linear_regression_comb.csv'
    save_prediction(y_pred, filename)
    # Mostramos los resultados
    print('Mejores hiperparámetros: ', grid.best_params_)


if __name__ == '__main__':
    main(feature_eng=False)
    print('-' * 50)
    main(feature_eng=True)

