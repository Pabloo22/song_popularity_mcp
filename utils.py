import pandas as pd


def load_data(split: bool = True,
              exclude_id: bool = False,
              no_duplicates: bool = False,
              version: int = 1,
              preprocessed: bool = False,
              feature_selection: bool = False,
              agg: bool = False):

    if preprocessed:
        if version == 1:
            train = pd.read_csv('data/train_preprocessed.csv')
            X_test = pd.read_csv('data/test_preprocessed.csv')
        elif version == 2:
            train = pd.read_csv('data/train_preprocessed_v2.csv')
            X_test = pd.read_csv('data/test_preprocessed_v2.csv')
        else:
            raise ValueError('Version no válida')
    elif version == 1:
        train = pd.read_csv('data/train.csv') if not agg else pd.read_csv('data/train_agg.csv')
        X_test = pd.read_csv('data/test.csv')
    elif version == 2:
        train = pd.read_csv('data/train_v2.csv') if not agg else pd.read_csv('data/train_agg_v2.csv')
        X_test = pd.read_csv('data/test_v2.csv')
    elif version == 3:
        train = pd.read_csv('data/train_v3.csv')
        X_test = pd.read_csv('data/test_v3.csv')
    else:
        raise ValueError('Invalid version')

    if feature_selection:
        cols_to_use = ['contains_feat', 'instrumentalness', 'is_derived', 'loudness', 'tempo', 'danceability',
                       'song_duration_ms', 'audio_valence', 'speechiness', 'liveness', 'num_lowercase_words',
                       'language', 'sentiment', 'num_syllables_missing', 'num_syllables', 'time_signature']
        train = train[cols_to_use + ['song_popularity']]
        X_test = X_test[cols_to_use]

    if no_duplicates and version == 1:
        # Excluimos 'song_id' a la hora de eliminar duplicados
        train = train.drop_duplicates(subset=train.drop('song_id', axis=1).columns)

    if exclude_id and not feature_selection:
        train = train.drop('song_id', axis=1, errors='ignore')
        X_test = X_test.drop('song_id', axis=1, errors='ignore')

    if split:
        X_train = train.drop(['song_popularity'], axis=1)
        y_train = train['song_popularity']
        return X_train, y_train, X_test

    return train, X_test


def print_grid_results(grid):
    print('Score medio de la validación cruzada: ', grid.best_score_)
    print('Desviación estándar de la validación cruzada: ', grid.cv_results_['std_test_score'][grid.best_index_])
    print('Best params: ', grid.best_params_)
    print(grid.best_estimator_)


def save_prediction(y_pred, filename: str, memorize: bool = False):
    X_test = pd.read_csv('data/test.csv')

    submission = pd.DataFrame({
        'song_id': X_test['song_id'],
        'song_popularity': y_pred,
        'song_name': X_test['song_name']
    })

    if memorize:
        # Cambiamos las predicciones que ya están en el dataset de entrenamiento porque coincide el 'song_name'
        train_agg = pd.read_csv('data/train_agg.csv')
        for song_name, popularity in zip(train_agg['song_name'], train_agg['song_popularity']):
            submission.loc[submission['song_name'] == song_name, 'song_popularity'] = popularity

        # Eliminamos la columna 'song_name'
        submission = submission.drop('song_name', axis=1)

    submission.to_csv('predicciones/' + filename, index=False)


def get_feature_importaces(X_train, clf):
    # Creamos el Dataframe
    feature_importances = pd.DataFrame(clf.feature_importances_,
                                        index=X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)