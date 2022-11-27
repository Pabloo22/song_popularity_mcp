"""
En este experimento se utilizarán las variables previamente seleccionadas y se tratará de optimizar los
hiperparámetros de un Random Forest Regressor. Esta vez, la selección de hiperparámetros se hará mediante un
algoritmo genético para acelerar el proceso y realizar una búsqueda informada. En concreto, haremos uso de la
librería TPOT.
"""

from tpot import TPOTRegressor
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


from utils import load_data, save_prediction, print_grid_results


def main():
    np.random.seed(2)

    X_train, y_train, X_test = load_data(split=True, version=1, exclude_id=True, feature_selection=False)

    # Eliminamos 'song_name'
    X_train = X_train.drop('song_name', axis=1)
    X_test = X_test.drop('song_name', axis=1)

    cv = KFold(n_splits=10, shuffle=True)
    # parameters = {
    #     'n_estimators': [200],
    #     'max_features': range(1, len(X_train.columns)),
    #     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # }
    # config_dict = {
    #     'sklearn.ensemble.RandomForestRegressor': parameters
    # }
    # pipeline_optimizer = TPOTRegressor(generations=4,
    #                                    population_size=8,
    #                                    cv=cv,
    #                                    scoring='r2',
    #                                    verbosity=2,
    #                                    n_jobs=-1,
    #                                    config_dict=config_dict)

    grid = GridSearchCV(estimator=RandomForestRegressor(n_estimators=200),
                                      param_grid={'max_features': [3, 4, 5,6, 7]},
                                      scoring='r2',
                                      cv=cv,
                                      n_jobs=-1,
                                      verbose=10)

    grid.fit(X_train, y_train)

    # print('-' * 50)
    # print('r2 score en train: ', r2_score(y_train, pipeline_optimizer.predict(X_train)))
    # print('mae en train: ', mean_absolute_error(y_train, pipeline_optimizer.predict(X_train)))

    print_grid_results(grid)

    model = grid.best_estimator_
    model.fit(X_train, y_train)

    # Imprimimos el r2 score en train
    print('r2 score en train: ', model.score(X_train, y_train))
    print('r2 score en train: ', r2_score(y_train, model.predict(X_train)))

    y_pred = model.predict(X_test)

    # y_pred = pipeline_optimizer.predict(X_test)

    # Guardamos las predicciones
    save_prediction(y_pred, 'exp2_random_forest_v5.csv')


if __name__ == '__main__':
    main()
