import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
import seaborn as sns
from metrics import Metrics
import matplotlib.patches as mpatches


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
    # Identificar colunas numéricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Utilizar StandardScaler para normalizar os dados numéricos
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Utilizar OneHotEncoder para colunas categóricas
    if cat_cols:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])

        X_train = np.hstack([X_train[num_cols], X_train_cat])
        X_test = np.hstack([X_test[num_cols], X_test_cat])

    # Inicializar o LabelEncoder para os rótulos
    le = LabelEncoder()

    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
    if y_test.dtype == 'object':
        y_test = le.transform(y_test)  # usar o mesmo encoder para garantir uma codificação consistente

    return y_train, y_test, X_train, X_test, le

def calculate_metrics(y_test, y_pred, metrics_obj):
    metrics_obj.conf_matrix = confusion_matrix(y_test, y_pred)
    metrics_obj.TP = np.diag(metrics_obj.conf_matrix)
    metrics_obj.FP = np.sum(metrics_obj.conf_matrix, axis=0) - metrics_obj.TP
    metrics_obj.FN = np.sum(metrics_obj.conf_matrix, axis=1) - metrics_obj.TP
    metrics_obj.TN = np.sum(metrics_obj.conf_matrix) - (metrics_obj.FP + metrics_obj.FN + metrics_obj.TP)

    metrics_obj.calculated_accuracy = (metrics_obj.TP + metrics_obj.TN) / (metrics_obj.TP + metrics_obj.FP + metrics_obj.FN + metrics_obj.TN)
    metrics_obj.calculated_precision = metrics_obj.TP / (metrics_obj.TP + metrics_obj.FP)
    metrics_obj.calculated_recall = metrics_obj.TP / (metrics_obj.TP + metrics_obj.FN)
    metrics_obj.calculated_specificity = metrics_obj.TN / (metrics_obj.TN + metrics_obj.FP)
    metrics_obj.calculated_f1 = 2 * (metrics_obj.calculated_precision * metrics_obj.calculated_recall) / (
                metrics_obj.calculated_precision + metrics_obj.calculated_recall)

    # metrics.auc_roc = roc_auc_score(y_test, y_pred)
    # AUC-ROC não é diretamente aplicável para classificação multiclasse no scikit-learn
    # Para multiclasse, geralmente se calcula um AUC por classe (one-vs-all) e depois se faz a média
    return metrics_obj

def print_metrics(metrics):
    print(f'Confusion Matrix: \n{metrics.conf_matrix}')
    print(f'True Positives: {metrics.TP}')
    print(f'False Positives: {metrics.FP}')
    print(f'True Negatives: {metrics.TN}')
    print(f'False Negatives: {metrics.FN}')
    print(f'Total True and False Positives and Negatives: {metrics.TP + metrics.FP + metrics.FN + metrics.TN}')
    print(f'Accuracy (calculated): {metrics.calculated_accuracy}')
    print(f'Precision (calculated): {metrics.calculated_precision}')
    print(f'Recall (calculated): {metrics.calculated_recall}')
    print(f'Specificity (calculated): {metrics.calculated_specificity}')
    print(f'F1 Score (calculated): {metrics.calculated_f1}')

def save_metrics_to_json(metrics, filename="metrics.json"):
    with open(filename, 'w') as file:
        json.dump(metrics, file)


# Função para adicionar hachuras e anotações
hatches = ['///', '....', 'xxxx', '--', 'ooo', '@@', '$$', '%%']

def add_hatches_and_annotations(ax, values):
    for bar, hatch, value in zip(ax.patches, hatches, values):
        bar.set_hatch(3 * hatch)
        bar.set_edgecolor('black')
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() - 0.05,
            f'{value:.2f}',
            ha="center",
            fontsize=12,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round4')
        )


def plot_metrics_for_class(class_name, metrics_values, metrics_names, save_path="graphics"):
    plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    sns.barplot(x=metrics_names, y=metrics_values, color='white', edgecolor='black', ax=ax)
    ax.set_title(f'Metrics for the {class_name} class', fontsize=16)
    add_hatches_and_annotations(ax, metrics_values)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"metrics_for_{class_name}.png"))


def plot_combined_metrics(calculated_metrics, class_names, metrics_names, save_path="graphics"):
    plt.figure(figsize=(14, 4.5))

    for idx, metric_name in enumerate(metrics_names):
        ax = plt.subplot(1, 4, idx + 1)
        values_for_metric = calculated_metrics[idx]
        sns.barplot(x=class_names, y=values_for_metric, dodge=True, color='white', edgecolor='black', ax=ax)
        ax.set_title(metric_name, fontsize=16)
        add_hatches_and_annotations(ax, values_for_metric)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "combined_metrics.png"))



