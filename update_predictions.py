from utils import update_submission_with_memorize


def main():
    # Extraemos el nombre de los archivos de la carpeta predicciones
    # files = ['baseline_cv10_reproducible.csv', 'exp1_linear_regression_comb.csv', 'exp1_linear_regression_no_comb.csv',
    #          'exp2_random_forest.csv', 'exp2_random_forest_v2.csv', 'exp2_random_forest_v3.csv',
    #          'exp2_random_forest_v4.csv', 'exp2_random_forest_v5.csv', 'exp4_automl_v1_2.csv', 'exp4_automl_v4.csv',
    #          'exp4_automl_v4_explain.csv']

    files = ['exp4_automl']

    files_without_csv = [file.replace('.csv', '') for file in files]

    # Actualizamos los archivos de predicciones
    for file in files_without_csv:
        update_submission_with_memorize(file)


if __name__ == '__main__':
    main()
