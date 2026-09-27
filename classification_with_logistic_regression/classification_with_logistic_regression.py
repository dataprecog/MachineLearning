#==============================================================================
# 
#==============================================================================
# Description.......: Customer Churn Classification with Logistic Regression
# Author............: Data Precog
# Email.............: info@dataprecog.com
# Date..............: 2026.09.23
# Version...........: 1.1

#================================================================================
# 1. Imports
#================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score

#================================================================================
# 2. read data
#================================================================================
df = pd.read_csv("data/telco_churn.csv")

#================================================================================
# 3. Data check, and Feature Selection
#================================================================================
# check the data
print(df.describe())
print(df.sample(5))

# Feature selection and target variable
X = df.drop(columns=["Churn"])
y = df["Churn"]

#================================================================================
# 4. train test split
#================================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#================================================================================
# 5. model fitting (Logistic Regression Classifier)
#================================================================================
# Scale features and train logistic regression
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

#================================================================================
# 6. Predict
#================================================================================
y_pred = model.predict(X_test)

coefficients = pd.Series(
    model.named_steps["classifier"].coef_[0],
    index=X.columns
).sort_values(ascending=False)

print(coefficients)

coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.show()

#================================================================================
# 8. Evaluation
#================================================================================
y_pred_probabilities = model.predict_proba(X_test)
print(y_pred_probabilities[0:10])

print("Log Loss......:", log_loss(y_test, y_pred_probabilities))
print("Accuracy......:", accuracy_score(y_test, y_pred))

#================================================================================
# 9. Test on unseen data
#================================================================================
np.random.seed(42)
number_of_rows = 50

columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

X_new = pd.DataFrame({
    "gender": np.random.randint(0, 2, number_of_rows),
    "SeniorCitizen": np.random.randint(0, 2, number_of_rows),
    "Partner": np.random.randint(0, 2, number_of_rows),
    "Dependents": np.random.randint(0, 2, number_of_rows),
    "tenure": np.random.randint(1, 73, number_of_rows),
    "PhoneService": np.random.randint(0, 2, number_of_rows),
    "MultipleLines": np.random.randint(0, 3, number_of_rows),
    "InternetService": np.random.randint(0, 3, number_of_rows),
    "OnlineSecurity": np.random.randint(0, 3, number_of_rows),
    "OnlineBackup": np.random.randint(0, 3, number_of_rows),
    "DeviceProtection": np.random.randint(0, 3, number_of_rows),
    "TechSupport": np.random.randint(0, 3, number_of_rows),
    "StreamingTV": np.random.randint(0, 3, number_of_rows),
    "StreamingMovies": np.random.randint(0, 3, number_of_rows),
    "Contract": np.random.randint(0, 3, number_of_rows),
    "PaperlessBilling": np.random.randint(0, 2, number_of_rows),
    "PaymentMethod": np.random.randint(0, 4, number_of_rows),
    "MonthlyCharges": np.round(np.random.uniform(20, 120, number_of_rows), 2),
    "TotalCharges": np.round(np.random.uniform(20, 8000, number_of_rows), 2),
})

predictions = model.predict(X_new)
print(predictions)

#================================================================================
# EOF
#================================================================================
