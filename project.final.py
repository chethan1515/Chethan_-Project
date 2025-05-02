import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv(r'C:\Users\cheth\Downloads\test (2).csv')

# Define the target column name
target_column = 'Loan_Status'  # Replace with your actual target column name

# Define features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify columns with missing values in X
columns_with_missing_values = X.columns[X.isnull().any()]

# Initialize SimpleImputer with 'most_frequent' strategy for all columns
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)  # Create a new DataFrame for imputed data

# Encode non-numeric columns using LabelEncoder
label_encoders = {}
for col in range(X_imputed.shape[1]):
    if df[X.columns[col]].dtype == 'object':
        label_encoders[col] = LabelEncoder()
        X_imputed[:, col] = label_encoders[col].fit_transform(X_imputed[:, col])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
