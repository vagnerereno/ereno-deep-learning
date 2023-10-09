import os
import model
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = utils.load_data();
    print(X_train.columns, len(X_test.columns))
    y_train, y_test, X_train, X_test = utils.preprocess_data(X_train, y_train, X_test, y_test)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    history = model.model(X_train, y_train, X_test)
    print(history)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
