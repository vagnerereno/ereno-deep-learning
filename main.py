import lime
import pip
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib as seaborn
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import shap
from lime.lime_tabular import LimeTabularExplainer
import gc
import os
import model
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = utils.load_data();
    print(X_train.columns, len(X_test.columns))
    y_train, y_test, X_train, X_test = utils.tratar_dados(X_train, y_train, X_test, y_test)
    print(X_train.columns, len(X_test.columns))
    history = model.model(X_train, y_train, X_test)
    print(history)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
