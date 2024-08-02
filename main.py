import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Load datasets
fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

# Add labels: 1 for Fake and 0 for True
fake_data['label'] = 1
true_data['label'] = 0

# Combine datasets
data = pd.concat([fake_data, true_data])

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = data['text']
Y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Fit and transform the training data, then transform the test data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the LinearSVC model
clf = LinearSVC()

# Train the model
clf.fit(X_train_vectorized, Y_train)

# Evaluate the model
accuracy = clf.score(X_test_vectorized, Y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix and classification report
Y_pred = clf.predict(X_test_vectorized)
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# Evaluate the model using cross-validation
cross_val_scores = cross_val_score(clf, X_train_vectorized, Y_train, cv=5)
print(f"\nCross-validation scores: {cross_val_scores}")
print(f"Average cross-validation score: {np.mean(cross_val_scores)}")

# Function to check if an article is real or fake
def check_article(article):
    # Transform the new article using the same vectorizer
    article_vectorized = vectorizer.transform([article])

    # Predict the class of the new article
    prediction = clf.predict(article_vectorized)
    predicted_class = "REAL" if prediction[0] == 0 else "FAKE"

    # Show result in messagebox
    messagebox.showinfo("Prediction Result", f"The article is {predicted_class}.")

# Create the GUI
def create_gui():
    window = tk.Tk()
    window.title("News Article Classifier")

    # Create a label
    label = tk.Label(window, text="Enter the news article below:")
    label.pack(pady=10)

    # Create a scrolled text area for input
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=80, height=15)
    text_area.pack(pady=10)

    # Create a button to check the article
    check_button = tk.Button(window, text="Check", command=lambda: check_article(text_area.get("1.0", tk.END).strip()))
    check_button.pack(pady=10)

    # Run the application
    window.mainloop()
 
# Run the GUI application
create_gui()
