import os

import numpy as np

import model
import utils
from metrics import Metrics

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

    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)
    print("Unique classes in y_train:", np.unique(y_train))

    print("X_train Shape after preprocess:", X_train.shape)
    print("X_test Shape after preprocess:", X_test.shape)

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("Classes distribution in training dataset after preprocess:\n", dict(zip(unique_train, counts_train)))
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("Classes distribution in training dataset after preprocess:\n", dict(zip(unique_test, counts_test)))

    class_names = np.unique(y_train)
    num_classes = len(class_names)
    metrics = Metrics(num_classes)
    original_class_names = le.inverse_transform(class_names)

    _, y_pred = model.model(X_train, y_train, X_test) # Para ver o histórico, armazenar a variável history.
    metrics = utils.calculate_metrics(y_test, y_pred, metrics)
    utils.print_metrics(metrics)

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for i, class_name in enumerate(original_class_names):
        metrics_values = [metrics.calculated_accuracy[i], metrics.calculated_precision[i], metrics.calculated_recall[i], metrics.calculated_f1[i]]
        utils.plot_metrics_for_class(str(class_name), metrics_values, metrics_names)

    combined_metrics = [metrics.calculated_accuracy, metrics.calculated_precision, metrics.calculated_recall, metrics.calculated_f1]
    utils.plot_combined_metrics(combined_metrics, [str(class_name) for class_name in original_class_names], metrics_names)

    print('ou aq', len(metrics.calculated_accuracy), len(metrics.calculated_precision), len(metrics.calculated_recall), len(metrics.calculated_f1))
    print('ou entao aq', len(original_class_names))

