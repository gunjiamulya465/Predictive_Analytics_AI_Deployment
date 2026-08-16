
import pandas as pd

# Load Heart Disease dataset
df = pd.read_csv("Heart.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)
# Display all column names
print("\nColumns in the dataset:")
print(df.columns.tolist())

sklearn.preprocessing
# Separate input features and target variable

X = df.drop("target", axis=1)
y = df["target"]

print("\nInput Features:")
print(X.columns)

print("\nTarget:")
print(y.name)
# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Logistic Regression Model

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

print("\nLogistic Regression Accuracy:",
      logistic_accuracy)
# Decision Tree Model

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:",
      tree_accuracy)
# Random Forest Model

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

y_pred_forest = random_forest.predict(X_test)

forest_accuracy = accuracy_score(
    y_test,
    y_pred_forest
)

print("\nRandom Forest Accuracy:",
      forest_accuracy)

# Compare Model Accuracies

print("\n--- Model Comparison ---")

print("Logistic Regression:",
      logistic_accuracy)

print("Decision Tree:",
      tree_accuracy)

print("Random Forest:",
      forest_accuracy)

# Find the best model
accuracies = {
    "Logistic Regression": logistic_accuracy,
    "Decision Tree": tree_accuracy,
    "Random Forest": forest_accuracy
}

best_model = max(accuracies, key=accuracies.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", accuracies[best_model])

import joblib

# Save the best model
if best_model == "Logistic Regression":
    final_model = logistic_model
elif best_model == "Decision Tree":
    final_model = decision_tree
else:
    final_model = random_forest

joblib.dump(final_model, "heart_disease_model.pkl")

print("\nBest model saved successfully!")
