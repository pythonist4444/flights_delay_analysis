import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------
# Machine Learning

# 3 Predicting Flight Delays: Using Machine Learning
# 3.1 Create a machine learning model to predict whether a flight will have a delay of 15 minutes or more at departure
# 3.2 Create a machine learning model to predict whether a flight will have a delay of 15 minutes or more at arrival
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_path = os.path.join(current_dir, 'flights.csv')

# Load the dataset
df = pd.read_csv(csv_path)

df_cleaned = df.dropna(subset=['arr_delay','dep_delay'])


df_cleaned['delayed'] = (df_cleaned['dep_delay'] > 15).astype(int) # Create a new column to indicate whether the flight was delayed or not

# 1 = Delayed (15 minutes or more)
# 0 = Not Delayed (less than 15 minutes)

# Feature Engineering: Is the process of creating new features from existing data to improve the performance of machine learning models
# Select features for the model
features = ['month', 'day', 'hour','minute', 'carrier', 'origin', 'dest', 'distance',]
X = df_cleaned[features]
y = df_cleaned['delayed']

# Convert categorical variables to numerical variables (one-hot encoding)
X_encoded = pd.get_dummies(X, columns=['origin', 'dest', 'carrier'], drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from imblearn.over_sampling import SMOTE

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# # Apply SMOTE to oversample the minority class
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_encoded, y)



# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2,random_state=42, stratify=y) 
# Stratify to ensure that the proportion of delayed and non-delayed flights is the same in both train and test sets
# test_size=0.2 means that 20% of the data will be used for testing and 80% for training
print(f"Shape of training set: {x_train.shape}")
print(f"Shape of testing set: {x_test.shape}")

# Logistic Regression
model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# print("Logistic Regression with SMOTE:")
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


from sklearn.tree import DecisionTreeClassifier

# Decision Tree
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
print(classification_report(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")

# print("Decision Tree with SMOTE:")
# print(classification_report(y_test, y_pred_tree))
# print(confusion_matrix(y_test, y_pred_tree))
# print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")


# from imblearn.under_sampling import RandomUnderSampler

# # Apply undersampling to balance the dataset
# undersampler = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = undersampler.fit_resample(X_encoded, y)

# # Split the resampled data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# # Logistic Regression
# model = LogisticRegression(max_iter=2000)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print("Logistic Regression with Undersampling:")
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# # Decision Tree
# tree = DecisionTreeClassifier(max_depth=4)
# tree.fit(x_train, y_train)
# y_pred_tree = tree.predict(x_test)
# print("Decision Tree with Undersampling:")
# print(classification_report(y_test, y_pred_tree))
# print(confusion_matrix(y_test, y_pred_tree))
# print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")





# Feature Importance
importances = tree.feature_importances_
feature_names = X_encoded.columns
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(X_encoded.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")
    
# Drop features with zero importance
important_features = [feature_names[i] for i in indices if importances[i] > 0]

X_reduced = X_encoded[important_features] # Create a dataset with only the important features

# Re-train the model with reduced features
x_train, x_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
tree.fit(x_train, y_train)

# Evaluate the model again
y_pred_tree = tree.predict(x_test)
print(classification_report(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")


# Plot the feature importances of the forest
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Feature importances (Reduced Model)")
plt.bar(range(len(important_features)), importances[indices], align="center")
plt.xticks(range(len(important_features)), [important_features[i] for i in indices], rotation=90)
plt.xlim([-1, len(important_features)])
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------------------------------

# 4. What underlying factors influence flight delays the most? Are some routes more prone to disruptions than others? Do external variables like time of day, distance, or carrier policies play a significant role? By analyzing the relationships between different features, you might discover unexpected insights.
