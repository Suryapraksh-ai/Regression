# Regression
🧠 Customer Churn Prediction using Artificial Neural Network (ANN)
🔍 Project Overview
This project is a practical implementation of a binary classification problem using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The goal is to predict whether a customer will churn (exit) from a bank, based on various features like credit score, geography, age, balance, and more.

The dataset used is from a real-world banking context, containing 10,000+ records.

🧰 Tech Stack
Python 3

TensorFlow / Keras for deep learning

Pandas / NumPy for data manipulation

Scikit-learn for preprocessing and evaluation

Matplotlib / Seaborn (optional for visualization)

📁 Dataset Description
The dataset includes the following features:

Geography (France, Germany, Spain)

Gender

Credit Score

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

Target variable:

Exited (1 = Churned, 0 = Stayed)

⚙️ How the Model Works
Data Preprocessing:

Label Encoding for gender

One-Hot Encoding for geography

Feature scaling using StandardScaler

Train/Test Split:

80% training / 20% test data

Model Architecture:

Input layer based on preprocessed features

2 Hidden layers with ReLU activation

Output layer with sigmoid activation (binary classification)

Compilation & Training:

Optimizer: adam

Loss Function: binary_crossentropy

Metric: accuracy

Trained over 100 epochs

Model Evaluation:

Predictions on test set

Confusion matrix & accuracy score

✅ Results
Confusion Matrix: Printed after prediction

Accuracy Score: Printed using accuracy_score from sklearn

✔️ Model successfully classifies customer churn with good accuracy using only structured tabular data.

