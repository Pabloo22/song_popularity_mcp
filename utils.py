import pandas as pd


def load_data(split: bool = True):
    train = pd.read_csv('data/train.csv')
    X_test = pd.read_csv('data/test.csv')

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