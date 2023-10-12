import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Step 1: Data Preprocessing
print("Starting Data Preprocessing...")

# Load data
data = pd.read_csv('data/preprocessed.csv')

# Drop unnecessary columns
columns_to_drop = ['Srno', 'PrepTime', 'CookTime',
                   'TotalTime', 'Servings', 'Cuisine', 'Course', 'Diet']
data_cleaned = data.drop(columns=columns_to_drop)

# Apply TF-IDF on Ingredients
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(data_cleaned['Ingredients'])

# Save transformed data and TF-IDF vectorizer
transformed_data = pd.DataFrame(
    X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
transformed_data['Recipe'] = data_cleaned['Recipe']
transformed_data.to_csv('data/transformed_data.csv', index=False)
joblib.dump(tfidf_vectorizer, 'features/tfidf_features.pkl')

print("Data Preprocessing Completed.")

# Step 2: Dimensionality Reduction
print("Starting Dimensionality Reduction...")

# Apply PCA
pca = PCA(n_components=1000)
X_pca = pca.fit_transform(X_tfidf.toarray())
pca_df = pd.DataFrame(X_pca)
pca_df['Recipe'] = data_cleaned['Recipe']
pca_df.to_csv('data/pca_transformed_data.csv', index=False)

print("Dimensionality Reduction Completed.")

# Step 3: Model Training
print("Starting Model Training...")

# Load PCA transformed data
data = pd.read_csv('data/pca_transformed_data.csv')
X = data.drop(columns=['Recipe'])
y = data['Recipe']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
joblib.dump(dt_classifier, 'models/decision_tree_model.pkl')

print('Decision Tree Training Completed.')

# Train KNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
joblib.dump(knn_classifier, 'models/knn_model.pkl')

print('KNN Training Completed.')

# Train Random Forests
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, 'models/random_forest_model.pkl')

print('Random Forest Training Completed.')

# Train Gradient Boosted Decision Trees
gbt_classifier = GradientBoostingClassifier(random_state=42)
gbt_classifier.fit(X_train, y_train)
joblib.dump(gbt_classifier, 'models/gradient_boosted_tree_model.pkl')

print('Gradient Boosted Tree Training Completed.')
print("Model Training Completed.")

# Step 4: Model Evaluation
print("Starting Model Evaluation...")

# Evaluate models on training data
dt_train_accuracy = accuracy_score(y_train, dt_classifier.predict(X_train))
knn_train_accuracy = accuracy_score(y_train, knn_classifier.predict(X_train))
rf_train_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train))
gbt_train_accuracy = accuracy_score(y_train, gbt_classifier.predict(X_train))

# Evaluate models on testing data
dt_test_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
knn_test_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))
rf_test_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
gbt_test_accuracy = accuracy_score(y_test, gbt_classifier.predict(X_test))

# Save metrics
with open('metrics/model_metrics.txt', 'w') as file:
    file.write(f"Decision Tree Training Accuracy: {dt_train_accuracy}\n")
    file.write(f"Decision Tree Testing Accuracy: {dt_test_accuracy}\n")
    file.write(f"KNN Training Accuracy: {knn_train_accuracy}\n")
    file.write(f"KNN Testing Accuracy: {knn_test_accuracy}\n")
    file.write(f"Random Forest Training Accuracy: {rf_train_accuracy}\n")
    file.write(f"Random Forest Testing Accuracy: {rf_test_accuracy}\n")
    file.write(
        f"Gradient Boosted Tree Training Accuracy: {gbt_train_accuracy}\n")
    file.write(
        f"Gradient Boosted Tree Testing Accuracy: {gbt_test_accuracy}\n")

print("Model Evaluation Completed.")
print("All tasks have been successfully completed!")
