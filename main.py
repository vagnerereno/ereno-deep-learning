import os

import numpy as np

import model
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = utils.load_data();

    print("X_train Columns:", X_train.columns, len(X_train.columns))
    print("X_test Columns:", X_test.columns, len(X_test.columns))
    print("X_train Head: ", X_train.head())
    print("X_test Head: ", X_test.head())
    print("Classes distribution in training dataset:\n", y_train.value_counts())
    print("Classes distribution in testing dataset:\n", y_test.value_counts())

    y_train, y_test, X_train, X_test = utils.preprocess_data(X_train, y_train, X_test, y_test)
    print("Unique classes in y_train:", np.unique(y_train))

    print("X_train Shape after preprocess:", X_train.shape)
    print("X_test Shape after preprocess:", X_test.shape)
    print("X_train[:5] after preprocess: ", X_train[:5])
    print("X_test[:5] after preprocess: ", X_test[:5])

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("Classes distribution in training dataset after preprocess:\n", dict(zip(unique_train, counts_train)))
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("Classes distribution in training dataset after preprocess:\n", dict(zip(unique_test, counts_test)))

    _, y_pred = model.model(X_train, y_train, X_test) # Para ver o histórico, armazenar a variável history.
    metrics = utils.calculate_metrics(y_test, y_pred)
    utils.print_metrics(metrics)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
