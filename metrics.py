class Metrics:
    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.conf_matrix = []
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