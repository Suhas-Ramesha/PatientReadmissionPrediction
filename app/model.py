# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Step 2: Load the dataset
df = pd.read_csv('./data/diabetes_data.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Step 3: Data Cleaning
# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Handle missing values for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Handle missing values for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Drop duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Step 4: Exploratory Data Analysis (EDA)
# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Distribution of the target variable (readmitted)
plt.figure(figsize=(8, 6))
sns.countplot(x='readmitted', data=df)
plt.title('Distribution of Readmission')
plt.show()

# Correlation matrix (only for numeric columns)
plt.figure(figsize=(12, 8))
corr_matrix = df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for selected numeric features (exclude 'readmitted')
sns.pairplot(df[['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_diagnoses']])
plt.show()

# Step 5: Feature Engineering
# Separate the target variable before encoding
y = df['readmitted']
X = df.drop('readmitted', axis=1)

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build a Predictive Model
# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Feature Importance
# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances')
plt.show()

# Step 8: Save the Model (Optional)
# Save the model to a file
joblib.dump(model, 'diabetes_readmission_model.pkl')

# Step 9: Load the Model and Make Predictions (Optional)
# Load the model from the file
loaded_model = joblib.load('diabetes_readmission_model.pkl')

# Make predictions with the loaded model
new_predictions = loaded_model.predict(X_test)

# Evaluate the loaded model
print("\nConfusion Matrix (Loaded Model):")
print(confusion_matrix(y_test, new_predictions))

print("\nClassification Report (Loaded Model):")
print(classification_report(y_test, new_predictions))