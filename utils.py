import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)

from metrics import Metrics


def load_data():
    # Carregamento de dados
    train_df = pd.read_csv('data/train.csv', sep=',')
    test_df = pd.read_csv('data/test.csv', sep=',')

    # Colunas enriquecidas para remover
    columns_to_remove = ['stDiff', 'sqDiff', 'gooseLenghtDiff', 'cbStatusDiff', 'apduSizeDiff',
                         'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange',
                         'delay', 'isbARms', 'isbBRms', 'isbCRms', 'ismARms', 'ismBRms', 'ismCRms',
                         'ismARmsValue', 'ismBRmsValue', 'ismCRmsValue', 'csbArms', 'csvBRms',
                         'csbCRms', 'vsmARms', 'vsmBRms', 'vsmCRms', 'isbARmsValue', 'isbBRmsValue',
                         'isbCRmsValue', 'vsbARmsValue', 'vsbBRmsValue', 'vsbCRmsValue',
                         'vsmARmsValue', 'vsmBRmsValue', 'vsmCRmsValue', 'isbATrapAreaSum',
                         'isbBTrapAreaSum', 'isbCTrapAreaSum', 'ismATrapAreaSum', 'ismBTrapAreaSum',
                         'ismCTrapAreaSum', 'csvATrapAreaSum', 'csvBTrapAreaSum', 'vsbATrapAreaSum',
                         'vsbBTrapAreaSum', 'vsbCTrapAreaSum', 'vsmATrapAreaSum', 'vsmBTrapAreaSum',
                         'vsmCTrapAreaSum', 'gooseLengthDiff']

    # Remoção de colunas enriquecidas e com NaN
    train_df = train_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')
    test_df = test_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')

    # Separação de features e labels
    X_train = train_df.drop(columns=['@class@'])
    y_train = train_df['@class@']
    X_test = test_df.drop(columns=['@class@'])
    y_test = test_df['@class@']
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):

    # Tratar valores faltantes
    for col in X_train.select_dtypes(include=[np.number]).columns.tolist():
        median_value = X_train[col].median()
        X_train[col].fillna(median_value, inplace=True)
        X_test[col].fillna(median_value, inplace=True)

    for col in X_train.select_dtypes(include=['object']).columns.tolist():
        X_train[col].fillna("missing", inplace=True)
        X_train[col] = X_train[col].astype(str)

        X_test[col].fillna("missing", inplace=True)
        X_test[col] = X_test[col].astype(str)

    # Identificar colunas numéricas e categóricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Inicializar o OneHotEncoder e o ColumnTransformer
    onehotencoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', onehotencoder, cat_cols)],
        remainder='passthrough')

    # Aplicar a transformação nos datasets de treino e teste
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Inicializar o LabelEncoder
    le = LabelEncoder()

    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
    if y_test.dtype == 'object':
        y_test = le.transform(y_test)

    return y_train, y_test, X_train, X_test

def calculate_metrics(y_test, y_pred):
    metrics = Metrics()

    metrics.accuracy = accuracy_score(y_test, y_pred)
    metrics.precision = precision_score(y_test, y_pred, average='weighted') # Para problemas multiclass, adicionar o argumento average
    metrics.recall = recall_score(y_test, y_pred, average='weighted') # Para problemas multiclass, adicionar o argumento average
    metrics.f1 = f1_score(y_test, y_pred, average='weighted') # Para problemas multiclass, adicionar o argumento average

    # AUC-ROC não é diretamente aplicável para classificação multiclasse no scikit-learn
    # Para multiclasse, geralmente se calcula um AUC por classe (one-vs-all) e depois se faz a média
    # metrics.auc_roc = roc_auc_score(y_test, y_pred)

    metrics.conf_matrix = confusion_matrix(y_test, y_pred)
    metrics.TP = np.diag(metrics.conf_matrix)
    metrics.FP = np.sum(metrics.conf_matrix, axis=0) - metrics.TP
    metrics.FN = np.sum(metrics.conf_matrix, axis=1) - metrics.TP
    metrics.TN = np.sum(metrics.conf_matrix) - (metrics.FP + metrics.FN + metrics.TP)

    metrics.calculated_accuracy = (metrics.TP + metrics.TN) / (metrics.TP + metrics.FP + metrics.FN + metrics.TN)
    metrics.calculated_precision = metrics.TP / (metrics.TP + metrics.FP)
    metrics.calculated_recall = metrics.TP / (metrics.TP + metrics.FN)
    metrics.calculated_specificity = metrics.TN / (metrics.TN + metrics.FP)
    metrics.calculated_f1 = 2 * (metrics.calculated_precision * metrics.calculated_recall) / (
                metrics.calculated_precision + metrics.calculated_recall)

    return metrics

def print_metrics(metrics):
    print(f'Accuracy: {metrics.accuracy}')
    print(f'Precision: {metrics.precision}')
    print(f'Recall: {metrics.recall}')
    print(f'F1 Score: {metrics.f1}')
    print(f'Confusion Matrix: \n{metrics.conf_matrix}')
    print(f'True Positives: {metrics.TP}')
    print(f'False Positives: {metrics.FP}')
    print(f'True Negatives: {metrics.TN}')
    print(f'False Negatives: {metrics.FN}')
    print(f'Accuracy (calculated): {metrics.calculated_accuracy}')
    print(f'Precision (calculated): {metrics.calculated_precision}')
    print(f'Recall (calculated): {metrics.calculated_recall}')
    print(f'Specificity (calculated): {metrics.calculated_specificity}')
    print(f'F1 Score (calculated): {metrics.calculated_f1}')