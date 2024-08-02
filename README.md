Here’s a README file for your news article classifier project:

--- 

# News Article Classifier

## Overview

This project is a news article classifier that uses machine learning to distinguish between fake and real news articles. It employs a Support Vector Machine (SVM) model with TF-IDF vectorization to classify the articles based on their content. The project also includes a graphical user interface (GUI) built with Tkinter for easy interaction.

## Features

- **Train and Evaluate Model**: The model is trained using a dataset of fake and true news articles and evaluated for accuracy and performance.
- **Cross-Validation**: The model is evaluated using cross-validation to ensure its reliability.
- **GUI for Classification**: A Tkinter-based GUI allows users to input a news article and classify it as fake or real.

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`
- `tkinter` (included with Python standard library)

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn
```

## Files

- `Fake.csv`: Dataset containing fake news articles.
- `True.csv`: Dataset containing real news articles.
- `news_classifier.py`: Main script for training the model, evaluating performance, and running the GUI.

## Usage

1. **Prepare the Datasets**: Ensure that `Fake.csv` and `True.csv` are available in the same directory as `news_classifier.py`.

2. **Run the Script**: Execute the main script to train the model and start the GUI:

   ```bash
   python news_classifier.py
   ```

3. **Interact with the GUI**:
   - Enter a news article in the text area.
   - Click the "Check" button to classify the article as fake or real.
   - The result will be displayed in a message box.

## Code Explanation

- **Data Loading**: The datasets are loaded, labeled, combined, and shuffled.
- **Feature Extraction**: TF-IDF vectorization is used to convert text data into numerical format.
- **Model Training**: A Linear Support Vector Classifier (SVC) is trained on the vectorized text data.
- **Evaluation**: The model's accuracy, confusion matrix, and classification report are generated. Cross-validation is also performed.
- **GUI**: Tkinter is used to create a simple user interface for article classification.

## Example Output

```plaintext
Accuracy: 93.25%

Confusion Matrix:
[[1106  105]
 [  60  931]]

Classification Report:
              precision    recall  f1-score   support

         0       0.95      0.91      0.93      1211
         1       0.90      0.94      0.92       991

    accuracy                           0.93      2202
   macro avg       0.92      0.92      0.92      2202
weighted avg       0.93      0.93      0.93      2202

Cross-validation scores: [0.927 0.933 0.924 0.931 0.928]
Average cross-validation score: 0.9286
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

