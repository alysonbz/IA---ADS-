from src.utils import process_diabetes


class Metrics:

    def __init__(self, y_pred, y_test):
        self.vp_c1 = 0
        self.vn_c1 = 0
        self.fp_c1 = 0
        self.fn_c1 = 0
        self.vp_c0 = 0
        self.vn_c0 = 0
        self.fp_c0 = 0
        self.fn_c0 = 0

    def set_param_classe1(self):

        for yp, yt in zip(y_pred, y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 = self.vp_c1 + 1
            if yp == 1 and yt == 0:
                self.fp_c1 = self.fp_c1 + 1
            if yp == 0 and yt == 1:
                self.fn_c1 = self.fn_c1 + 1
            if yp == 0 and yt == 0:
                self.vn_c1 = self.vn_c1 + 1

    def set_param_classe2(self):
        pass

    def compute_acuraccy(self):
        return None

    def compute_recall_c1(self):
        return None

    def compute_recall_c0(self):
        return None

    def compute_precision_c1(self):
        return None

    def compute_precision_c0(self):
        return None

    def compute_f1_score_c1(self):
        return None

    def compute_f1_score_c0(self):
        return None

    def compute_confusion_matriz(self):
        return None


y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe2()

print("acurácia geral:", mt.compute_acuraccy())
#
print("recall classe 0: ")
#
print("recall classe 1: ")
#
print("precision classe 0: ")
#
print("precision classe 1:")
#
print("F1-score classe 0:")
#
print("F1-score classe 1:")
#
print("Matriz de confusão")
#