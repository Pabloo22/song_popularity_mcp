import pandas as pd


def _get_filename(agg):
    if agg == 0:
        filename = 'train'
    elif agg == 1:
        filename = 'train_agg'
    elif agg == 2:
        filename = 'train_agg2'
    else:
        raise ValueError('agg no válido')

    return filename


def load_data(split: bool = True,
              exclude_id: bool = False,
              no_duplicates: bool = False,
              version: int = 1,
              preprocessed: bool = False,
              feature_selection: bool = False,
              agg: int = 0):
    filename = _get_filename(agg)
    if preprocessed:
        if version == 1:
            train = pd.read_csv(f'data/{filename}_preprocessed.csv')
            X_test = pd.read_csv('data/test_preprocessed.csv')
        elif version == 2:
            train = pd.read_csv(f'data/{filename}_preprocessed_v2.csv')
            X_test = pd.read_csv('data/test_preprocessed_v2.csv')
        else:
            raise ValueError('Version no válida')
    elif version == 1:
        train = pd.read_csv(f'data/{filename}.csv')
        X_test = pd.read_csv('data/test.csv')
    elif version == 2:
        filename = _get_filename(agg)
        train = pd.read_csv(f'data/{filename}_v2.csv')
        X_test = pd.read_csv('data/test_v2.csv')
    elif version == 3:
        train = pd.read_csv(f'data/{filename}_v3.csv')
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
        train_agg = pd.read_csv('data/memorize_dataset.csv')
        for song_name, popularity in zip(train_agg['song_name'], train_agg['song_popularity']):
            submission.loc[submission['song_name'] == song_name, 'song_popularity'] = popularity

        # Eliminamos la columna 'song_name'
        submission = submission.drop('song_name', axis=1)

    submission.to_csv('predicciones/' + filename, index=False)


def update_submission_with_memorize(filename: str):
    _, X_test = load_data(split=False)
    submission = pd.read_csv(f'predicciones/{filename}.csv')
    # Submission contiene una columna 'song_id' y otra 'song_popularity', por eso es necesario hacer un merge
    # con el dataset de test para obtener la columna 'song_name'
    submission = submission.merge(X_test[['song_id', 'song_name']], on='song_id', how='left')

    train_agg = pd.read_csv('data/memorize_dataset.csv')
    for song_name, popularity in zip(train_agg['song_name'], train_agg['song_popularity']):
        submission.loc[submission['song_name'] == song_name, 'song_popularity'] = popularity

    submission = submission.drop('song_name', axis=1)

    submission.to_csv(f'predicciones/{filename}_updated.csv', index=False)
