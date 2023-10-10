import numpy as np


class Metrics:
    def __init__(self, num_classes):
        self.accuracy = [0] * num_classes
        self.precision = [0] * num_classes
        self.recall = [0] * num_classes
        self.f1 = [0] * num_classes
        self.conf_matrix = np.zeros((num_classes, num_classes))
        self.TP = []
        self.FP = []
        self.TN = []
        self.FN = []
        self.calculated_accuracy = []
        self.calculated_precision = []
        self.calculated_recall = []
        self.calculated_specificity = []
        self.calculated_f1 = []
        self.auc_roc = 0

    def get_values_for_class(self, class_index):
        return [self.accuracy[class_index], self.precision[class_index], self.recall[class_index], self.f1[class_index]]
