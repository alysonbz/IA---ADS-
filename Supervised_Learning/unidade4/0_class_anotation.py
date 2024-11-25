from src.utils import load_diabetes_clean_dataset
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

diabetes_df = load_diabetes_clean_dataset()
print(diabetes_df.head())

X = diabetes_df.drop(["glucose", "diastolic"], axis=1).values
y = diabetes_df["glucose"].values

print(type(X), type(y))

X_bmi = X[:, 3]
print(y.shape, X_bmi.shape)

X_bmi = X_bmi.reshape(-1, 1)

print(X_bmi.shape)

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dL)")
plt.xlabel("Body Mass Index")
plt.show()



