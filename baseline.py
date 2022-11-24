import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_data, save_prediction


def main():
    np.random.seed(2)

    # Cargamos los datos
    train, X_test = load_data(split=False)

    # Separamos las variables predictoras de la variable objetivo y eliminamos 'song_name'
    X_train = train.drop(['song_id', 'song_popularity', 'song_name'], axis=1)
    y_train = train['song_popularity']

    # Creamos el pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNet())
    ])

    # Creamos el grid search
    alphas = np.arange(0.001, 0.1, 0.025)
    l1_ratios = np.arange(0.0001, 0.01, 0.025)
    grid = GridSearchCV(pipe, param_grid={'elasticnet__alpha': alphas,
                                          'elasticnet__l1_ratio': l1_ratios},
                        scoring='r2', cv=KFold(10, shuffle=True), n_jobs=-1, verbose=1)

    grid.fit(X_train, y_train)

    print('Score medio de la validación cruzada: ', grid.best_score_)
    print('Desviación estándar de la validación cruzada: ', grid.cv_results_['std_test_score'][grid.best_index_])

    # Obtenemos los mejores hiperparámetros
    best_params = grid.best_params_
    print('Best params: ', best_params)
    print(pipe)

    # Entrenamos el modelo con todos los datos
    model = grid.best_estimator_
    model.fit(X_train, y_train)

    # Predecimos con los datos de test
    y_pred = model.predict(X_test.drop(['song_id', 'song_name'], axis=1))

    save_prediction(y_pred, 'baseline_cv10_reproducible.csv')


if __name__ == '__main__':
    main()
