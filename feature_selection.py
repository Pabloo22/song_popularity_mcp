from featurewiz import featurewiz

from utils import load_data


def main():

    train, X_test = load_data(split=False, preprocessed=True, exclude_id=True)

    features = featurewiz(train, target='song_popularity', corr_limit=0.7, verbose=2)

    print(features)


if __name__ == '__main__':
    main()
