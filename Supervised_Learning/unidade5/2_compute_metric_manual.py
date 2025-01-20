from src.utils import process_diabetes

class Metrics:
    def __init__(self, y_pred, y_test):
        self.vp_c1 = 0  # Verdadeiros Positivos classe 1
        self.vn_c1 = 0  # Verdadeiros Negativos classe 1
        self.fp_c1 = 0  # Falsos Positivos classe 1
        self.fn_c1 = 0  # Falsos Negativos classe 1
        self.vp_c0 = 0  # Verdadeiros Positivos classe 0
        self.vn_c0 = 0  # Verdadeiros Negativos classe 0
        self.fp_c0 = 0  # Falsos Positivos classe 0
        self.fn_c0 = 0  # Falsos Negativos classe 0
        self.y_pred = y_pred
        self.y_test = y_test

    def set_param_classe1(self):
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 = self.vp_c1 + 1
            if yp == 1 and yt == 0:
                self.fp_c1 = self.fp_c1 + 1
            if yp == 0 and yt == 1:
                self.fn_c1 = self.fn_c1 + 1
            if yp == 0 and yt == 0:
                self.vn_c1 = self.vn_c1 + 1

    def set_param_classe2(self):
        # Na classe 0 os valores são invertidos em relação à classe 1
        self.vp_c0 = self.vn_c1
        self.vn_c0 = self.vp_c1
        self.fp_c0 = self.fn_c1
        self.fn_c0 = self.fp_c1

    def compute_acuraccy(self):
        # (VP + VN) / (VP + VN + FP + FN)
        total = self.vp_c1 + self.vn_c1 + self.fp_c1 + self.fn_c1
        if total == 0:
            return 0
        return (self.vp_c1 + self.vn_c1) / total

    def compute_recall_c1(self):
        # VP / (VP + FN)
        if (self.vp_c1 + self.fn_c1) == 0:
            return 0
        return self.vp_c1 / (self.vp_c1 + self.fn_c1)

    def compute_recall_c0(self):
        # VP / (VP + FN) para classe 0
        if (self.vp_c0 + self.fn_c0) == 0:
            return 0
        return self.vp_c0 / (self.vp_c0 + self.fn_c0)

    def compute_precision_c1(self):
        # VP / (VP + FP)
        if (self.vp_c1 + self.fp_c1) == 0:
            return 0
        return self.vp_c1 / (self.vp_c1 + self.fp_c1)

    def compute_precision_c0(self):
        # VP / (VP + FP) para classe 0
        if (self.vp_c0 + self.fp_c0) == 0:
            return 0
        return self.vp_c0 / (self.vp_c0 + self.fp_c0)

    def compute_f1_score_c1(self):
        # 2 * (precision * recall) / (precision + recall)
        precision = self.compute_precision_c1()
        recall = self.compute_recall_c1()
        if (precision + recall) == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def compute_f1_score_c0(self):
        # 2 * (precision * recall) / (precision + recall) para classe 0
        precision = self.compute_precision_c0()
        recall = self.compute_recall_c0()
        if (precision + recall) == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def compute_confusion_matriz(self):
        return [
            [self.vp_c0, self.fn_c0],
            [self.fn_c1, self.vp_c1]
        ]

y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe2()

print("acurácia geral:", mt.compute_acuraccy())
print("recall classe 0:", mt.compute_recall_c0())
print("recall classe 1:", mt.compute_recall_c1())
print("precision classe 0:", mt.compute_precision_c0())
print("precision classe 1:", mt.compute_precision_c1())
print("F1-score classe 0:", mt.compute_f1_score_c0())
print("F1-score classe 1:", mt.compute_f1_score_c1())
print("Matriz de confusão")
for i in mt.compute_confusion_matriz():
    print(i)
