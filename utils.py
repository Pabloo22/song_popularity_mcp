import pandas as pd


def load_data(split: bool = True,
              exclude_id: bool = False,
              no_duplicates: bool = False,
              version: int = 1,
              preprocessed: bool = False,
              feature_selection: bool = False):

    if preprocessed:
        train = pd.read_csv('data/train_preprocessed.csv')
        X_test = pd.read_csv('data/test_preprocessed.csv')
    elif version == 1:
        train = pd.read_csv('data/train.csv')
        X_test = pd.read_csv('data/test.csv')
    elif version == 2:
        train = pd.read_csv('data/train_v2.csv')
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
        train = train.drop('song_id', axis=1)
        X_test = X_test.drop('song_id', axis=1)

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



def save_prediction(y_pred, filename: str):
    X_test = pd.read_csv('data/test.csv')
    submission = pd.DataFrame({
        'song_id': X_test['song_id'],
        'song_popularity': y_pred
    })
    submission.to_csv('predicciones/' + filename, index=False)
