import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


from utils import load_data, save_prediction, print_grid_results


def main(add_only_contains_feat=False):
    np.random.seed(2)

    X_train, y_train, X_test = load_data(split=True, version=2, exclude_id=True, preprocessed=True)

    if add_only_contains_feat:
        old_X_train, _, old_X_test = load_data(split=True, version=1, exclude_id=True)
        X_train = pd.concat([X_train['contains_feat'], old_X_train], axis=1)
        X_test = pd.concat([X_test['contains_feat'], old_X_test], axis=1)

    cv = KFold(n_splits=10, shuffle=True)

    grid = GridSearchCV(estimator=RandomForestRegressor(n_estimators=200),
                        param_grid={'max_features': [3, 4, 5, 6, 7, 8, 9, 10]},
                        scoring='r2',
                        cv=cv,
                        n_jobs=-1,
                        verbose=10)

    grid.fit(X_train, y_train)

    print_grid_results(grid)

    model = grid.best_estimator_
    model.fit(X_train, y_train)

    # Imprimimos el r2 score en train
    print('r2 score en train: ', model.score(X_train, y_train))

    y_pred = model.predict(X_test)

    save_prediction(y_pred, 'exp3_random_forest.csv')


if __name__ == '__main__':
    main(True)
