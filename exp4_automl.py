from supervised.automl import AutoML
from sklearn.metrics import r2_score

from utils import load_data, save_prediction


def main():

    X_train, y_train, X_test = load_data(split=True, version=1, exclude_id=True, no_duplicates=False)

    # Eliminamos 'song_name'
    X_train = X_train.drop(columns=["song_name"], errors="ignore")
    X_test = X_test.drop(columns=["song_name"], errors="ignore")

    automl = AutoML(mode="Explain",
                    results_path="automl_results_4_explain",
                    ml_task="regression",
                    random_state=2,
                    eval_metric="r2")

    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    print("train r2 score: ", r2_score(y_train, automl.predict(X_train)))
    save_prediction(y_pred, 'exp4_automl_v4_explain.csv')


if __name__ == '__main__':
    main()
