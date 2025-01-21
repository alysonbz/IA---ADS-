import matplotlib.pyplot as plt
from src.utils import log_reg_diabetes
from sklearn.metrics import roc_curve

y_prob, y_test, _ = log_reg_diabetes()

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')

# Plot tpr against fpr
plt.plot(fpr, tpr, label='Curva ROC')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Diabetes Prediction')
plt.legend()
plt.show()
