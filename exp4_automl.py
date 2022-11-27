from supervised.automl import AutoML
from sklearn.metrics import r2_score

from utils import load_data, save_prediction


def main():

    X_train, y_train, X_test = load_data(split=True, version=2, exclude_id=True, preprocessed=True)

    automl = AutoML(mode="Compete", results_path="automl_results", ml_task="regression", random_state=2)

    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    print("train r2 score: ", r2_score(y_train, automl.predict(X_train)))
    save_prediction(y_pred, 'exp4_automl.csv')


if __name__ == '__main__':
    main()
