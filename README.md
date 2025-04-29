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

#POLYNOMIAL
📈 Polynomial vs Linear Regression: Salary Prediction
🔍 Project Overview
This project demonstrates the difference between Linear Regression and Polynomial Regression using a simple but realistic dataset — employee position levels vs. salaries. It clearly shows how polynomial models can better capture non-linear relationships in data, especially when predicting real-world salary trends.

📊 Dataset Overview
The dataset contains the following columns:

Position: Job title (e.g., Jr Software Engineer, Manager)

Level: Numerical level (used for prediction)

Salary: Corresponding salary value


Position	Level	Salary
Jr Software Engineer	1	45,000
Sr Software Engineer	2	50,000
Team Lead	3	60,000
Manager	4	80,000
Sr Manager	5	110,000
...	...	...
CEO	10	1,000,000
🛠️ Tech Stack
Python 3

Pandas and NumPy for data manipulation

Matplotlib for data visualization

Scikit-learn for both regression models

📈 Model Implementation
✅ Linear Regression
Assumes a straight-line relationship between level and salary

Model trained using LinearRegression()

Prediction for level 6.5: $330,378

✅ Polynomial Regression (Degree = 2 by default)
Captures curvature in data more effectively

Transforms features using PolynomialFeatures

Trained using LinearRegression() on the polynomial features

Prediction for level 6.5: $189,498

📉 Visualization
🔵 Linear Regression

🔷 Polynomial Regression

🚨 Note: In this case, Polynomial Regression gives a more realistic prediction, as the salary vs. level relationship is not linear.

🤔 Why This Matters
Demonstrates underfitting vs. better fit

Shows practical usage of PolynomialFeatures

Highlights how simple feature engineering can dramatically improve predictions

💡 Use Cases
Salary forecasting

Career-level compensation planning

Understanding model bias vs. variance trade-offs



