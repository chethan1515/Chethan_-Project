from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv(r'D:\data set\loandata.csv')

# Assuming 'X' contains features and 'y' contains the target column
X = data.drop(columns=['Loan_Status', 'Loan_ID'])  # Drop Loan_ID assuming it's a categorical variable
y = data['Loan_Status']

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Preprocessing - Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Support Vector Machine (SVM) model
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred) * 100

# Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred) * 100

print("SVM Accuracy:", svc_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

# Accuracy scores
accuracy_scores = [lr_accuracy, svc_accuracy]
models = ['Logistic Regression', 'SVM']

# Plotting the bar graph
plt.figure(figsize=(8, 5))
plt.bar(models, accuracy_scores, color=['blue', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Logistic Regression vs. SVM')
plt.ylim(0, 100)
plt.show()
