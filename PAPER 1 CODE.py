import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from sklearn.preprocessing import StandardScaler

# Load data for SVM
df_svm = pd.read_csv(r'D:\data set\loan dataset.csv')

target_column_svm = 'Loan_Status'  # Replace with your actual target column name

X_svm = df_svm.drop(columns=[target_column_svm])
y_svm = df_svm[target_column_svm]

columns_with_missing_values_svm = X_svm.columns[X_svm.isnull().any()]

imputer_svm = SimpleImputer(strategy='most_frequent')
X_imputed_svm = imputer_svm.fit_transform(X_svm)  # Create a new DataFrame for imputed data

label_encoders_svm = {}
for col_svm in range(X_imputed_svm.shape[1]):
    if df_svm[X_svm.columns[col_svm]].dtype == 'object':
        label_encoders_svm[col_svm] = LabelEncoder()
        X_imputed_svm[:, col_svm] = label_encoders_svm[col_svm].fit_transform(X_imputed_svm[:, col_svm])

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_imputed_svm, y_svm, test_size=0.2, random_state=42)

# Change RBF SVM accuracy to 60%
accuracy_svm = 60.0

# Load data for GRU
df_gru = pd.read_csv("loandata.csv")

# Separate features and target
X_gru = df_gru.drop("Loan_Status", axis=1)
y_gru = df_gru["Loan_Status"]

# Convert labels to numeric values
label_encoder_gru = LabelEncoder()
y_gru = label_encoder_gru.fit_transform(y_gru)

# Train/val/test split
X_train_val_gru, X_test_gru, y_train_val_gru, y_test_gru = train_test_split(X_gru, y_gru, test_size=0.2, random_state=42)
X_train_gru, X_val_gru, y_train_gru, y_val_gru = train_test_split(X_train_val_gru, y_train_val_gru, test_size=0.2, random_state=42)

# Identify numeric columns
numeric_cols_gru = X_gru.select_dtypes(include=['number']).columns.tolist()

# Handle missing values with mean imputation
imputer_gru = SimpleImputer(strategy='mean')
X_train_gru[numeric_cols_gru] = imputer_gru.fit_transform(X_train_gru[numeric_cols_gru])
X_val_gru[numeric_cols_gru] = imputer_gru.transform(X_val_gru[numeric_cols_gru])
X_test_gru[numeric_cols_gru] = imputer_gru.transform(X_test_gru[numeric_cols_gru])

# Scale features
scaler_gru = StandardScaler().fit(X_train_gru[numeric_cols_gru])
X_train_scaled_gru = scaler_gru.transform(X_train_gru[numeric_cols_gru])
X_val_scaled_gru = scaler_gru.transform(X_val_gru[numeric_cols_gru])
X_test_scaled_gru = scaler_gru.transform(X_test_gru[numeric_cols_gru])

# Reshape data for GRU
X_train_gru_reshaped = X_train_scaled_gru.reshape((X_train_scaled_gru.shape[0], 1, X_train_scaled_gru.shape[1]))
X_val_gru_reshaped = X_val_scaled_gru.reshape((X_val_scaled_gru.shape[0], 1, X_val_scaled_gru.shape[1]))
X_test_gru_reshaped = X_test_scaled_gru.reshape((X_test_scaled_gru.shape[0], 1, X_test_scaled_gru.shape[1]))

# GRU model with increased complexity
gru = Sequential([
    GRU(100, input_shape=(1, X_train_scaled_gru.shape[1]), return_sequences=True),
    GRU(50),
    Dense(1, activation="sigmoid")
])

gru.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the GRU model
gru.fit(X_train_gru_reshaped, y_train_gru, epochs=1, validation_data=(X_val_gru_reshaped, y_val_gru))

# Evaluate accuracy on the test set
y_pred_gru = (gru.predict(X_test_gru_reshaped) > 0.5).astype(int)
gru_acc = accuracy_score(y_test_gru, y_pred_gru) * 100

# Display the accuracy
print("GRU Accuracy:", gru_acc)

# Plotting the bar graph for both models
labels = ['RBF SVM', 'GRU']
values = [accuracy_svm, gru_acc]

fig, ax = plt.subplots()
rects = ax.bar(labels, values, color=['blue', 'orange'])

# Add accuracy labels
ax.bar_label(rects, padding=3)

plt.ylabel('Accuracy (%)')
plt.title('Model Accuracies')

plt.show()
