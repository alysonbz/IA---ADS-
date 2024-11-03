import matplotlib.pyplot as plt
from src.utils import log_reg_diabetes
# Import roc_curve
____

y_prob,y_test ,_= log_reg_diabetes()

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = ____(____, ____)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(____, ____)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()