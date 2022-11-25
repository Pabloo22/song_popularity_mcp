import pandas as pd


def load_data(split: bool = True, exclude_id: bool = False, no_duplicates: bool = False):
    train = pd.read_csv('data/train.csv')
    X_test = pd.read_csv('data/test.csv')

    if no_duplicates:
        # Excluimos 'song_id' a la hora de eliminar duplicados
        train = train.drop_duplicates(subset=train.drop('song_id', axis=1).columns)

    if exclude_id:
        train = train.drop('song_id', axis=1)
        X_test = X_test.drop('song_id', axis=1)

    if split:
        X_train = train.drop(['song_popularity'], axis=1)
        y_train = train['song_popularity']
        return X_train, y_train, X_test

    return train, X_test


def save_prediction(y_pred, filename: str):
    X_test = pd.read_csv('data/test.csv')
    submission = pd.DataFrame({
        'song_id': X_test['song_id'],
        'song_popularity': y_pred
    })
    submission.to_csv('predicciones/' + filename, index=False)
