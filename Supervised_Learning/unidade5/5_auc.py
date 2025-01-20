from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

y_prob,y_test,y_pred = log_reg_diabetes()

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_prob))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))

