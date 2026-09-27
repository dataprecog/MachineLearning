#==============================================================================
# 
#==============================================================================
# Description.......: Simple Linear Regression using scikit-learn
# Author............: Data Precog
# Email.............: info@dataprecog.com
# Date..............: 2026.09.09
# Version...........: 1.0

#==============================================================================
# Imports
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#==============================================================================
# Read data
#==============================================================================
df = pd.read_csv("data/Advertising.csv")

#==============================================================================
# Check data and Feature Selection
#==============================================================================
print(df.describe())
print(df.sample(5))

X = df[["TV"]]
y = df["sales"]

#==============================================================================
# Train Test Split
#==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

#==============================================================================
# Fit model
#==============================================================================
model = LinearRegression()
model.fit(X_train, y_train)

#==============================================================================
# Predict
#==============================================================================
y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.show()

#==============================================================================
# Evaluate
#==============================================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

slope = model.coef_[0]
intercept = model.intercept_

print("MAE.................: ", mae)
print("MSE.................: ", mse)
print("RMSE................: ", rmse)
print("R2..................: ", r2)

print("slope...............: ", slope)
print("intercept...........: ", intercept)

#==============================================================================
# Test on unseen data
#==============================================================================
X_new = pd.DataFrame({"TV": np.arange(50, 400)})
y_new_pred = model.predict(X_new)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_new, y_new_pred, color='red')
plt.show()

#==============================================================================
# EOF
#==============================================================================
