# Logistic Regression Binary Classification

## Objective
Build a binary classifier using logistic regression to classify instances into two categories using the Breast Cancer Wisconsin dataset.

## Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset from the scikit-learn library. It contains features computed from digitized images of breast mass and classifies tumors as malignant or benign.

## Approach
- Load the dataset from scikit-learn.
- Split the dataset into training and testing sets (70% train, 30% test).
- Standardize the feature values for better performance.
- Train a logistic regression model on the training data.
- Predict the class labels on test data.
- Evaluate model performance based on confusion matrix, precision, recall, and ROC-AUC score.

## Key Concepts
- **Logistic Regression:** A statistical model used for binary classification that predicts the probability of a class using a logistic function (sigmoid).
- **Sigmoid Function:** Converts linear regression output to a probability between 0 and 1.
- **Threshold Tuning:** Adjusting the decision boundary probability threshold to optimize classification results.
- **Evaluation Metrics:**
  - Confusion Matrix: A table showing true positives, false positives, true negatives, and false negatives.
  - Precision: Ratio of correctly predicted positive observations to total predicted positives.
  - Recall: Ratio of correctly predicted positive observations to all actual positives.
  - ROC-AUC Score: Measures the ability of the model to distinguish between classes.

## How to Run
1. Ensure Python and required libraries (`scikit-learn`, `pandas`, `matplotlib`) are installed.
2. Run the script: `python logistic_regression.py`.
3. The output will display evaluation metrics.

## Results
- Confusion Matrix, Precision, Recall, and ROC-AUC score values will be printed to assess the model's performance.

## Author
Adithya

## License
This project is for educational purposes.
