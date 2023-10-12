import tkinter as tk
from tkinter import ttk, messagebox
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Load trained models and features
models = {
    "Decision Tree": joblib.load('project_root/models/decision_tree_model.pkl'),
    "KNN": joblib.load('project_root/models/knn_model.pkl'),
    "Random Forest": joblib.load('project_root/models/random_forest_model.pkl'),
    "Gradient Boosted Tree": joblib.load('project_root/models/gradient_boosted_tree_model.pkl')
}
tfidf_vectorizer = joblib.load('project_root/features/tfidf_features.pkl')


def predict_recipe():
    ingredients = ingredients_entry.get()
    selected_model = model_var.get()
    model = models[selected_model]

    # Transforming user input
    transformed_ingredients = tfidf_vectorizer.transform([ingredients])

    # Predict using the model
    prediction = model.predict(transformed_ingredients)

    # Display the prediction
    prediction_var.set(prediction[0])


def load_metrics():
    with open('project_root/metrics/model_metrics.txt', 'r') as file:
        lines = file.readlines()
    return [line.split(':') for line in lines]


# GUI Setup
root = tk.Tk()
root.title("Recipe Predictor")

# Title
ttk.Label(root, text="Thomas Cox Seng 609 Final",
          font=("Arial", 16)).pack(pady=20)

# Ingredients Entry
ttk.Label(root, text="Enter Ingredients:", font=("Arial", 12)).pack(pady=10)
ingredients_entry = ttk.Entry(root, width=50)
ingredients_entry.pack(pady=10, padx=20)

# Model Dropdown
model_var = tk.StringVar(root)
model_var.set("Decision Tree")
ttk.Label(root, text="Select Model:", font=("Arial", 12)).pack(pady=10)
model_dropdown = ttk.OptionMenu(root, model_var, *models.keys())
model_dropdown.pack(pady=10, padx=20)

# Predict Button
predict_btn = ttk.Button(root, text="Predict Recipe", command=predict_recipe)
predict_btn.pack(pady=20)

# Prediction Display
prediction_var = tk.StringVar(root)
prediction_entry = ttk.Entry(
    root, width=50, state="readonly", textvariable=prediction_var)
prediction_entry.pack(pady=10, padx=20)

# Metrics Display in Treeview
ttk.Label(root, text="Model Metrics", font=("Arial", 12)).pack(pady=10)
metrics_tree = ttk.Treeview(root, columns=(
    "Model", "Accuracy"), show="headings")
metrics_tree.heading("Model", text="Model")
metrics_tree.heading("Accuracy", text="Accuracy")
metrics_tree.pack(pady=20, padx=20, fill="both", expand=True)
for model, accuracy in load_metrics():
    metrics_tree.insert("", "end", values=(model.strip(), accuracy.strip()))

# Instructions
instructions = """Instructions:
- Enter the ingredients separated by commas.
- Select the desired prediction model from the dropdown.
- Click 'Predict Recipe' to get the prediction."""
ttk.Label(root, text=instructions, wraplength=400,
          font=("Arial", 10)).pack(pady=20, padx=20)

root.mainloop()
