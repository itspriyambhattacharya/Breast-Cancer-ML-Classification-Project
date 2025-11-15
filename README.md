# 🩺 Breast Cancer Classification using Machine Learning

A Machine Learning Project using **Decision Tree Classifier** on the
**Breast Cancer Wisconsin Dataset** (Scikit-Learn)

## 📌 Overview

This project implements a complete **end-to-end machine learning
workflow** for classifying breast cancer tumors as **Malignant (0)** or
**Benign (1)** using the **Decision Tree Classifier**.

The project demonstrates: - Loading and exploring a real-world dataset -
Pre-processing using **StandardScaler** - Training a Decision Tree
Classifier - Performance evaluation using accuracy, confusion matrix &
classification report - Visual analysis using **Matplotlib** and
**Seaborn**

## 📁 Repository Structure

    Breast-Cancer-ML-Classification-Project/
    │
    ├── classification_breast_cancer.ipynb
    ├── main.py
    ├── README.md
    └── requirements.txt

## 📊 Dataset Information

- **Total Samples:** 569
- **Features:** 30
- **Classes:** Malignant (0), Benign (1)

## 🚀 Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## 🧠 Machine Learning Workflow

1.  Load dataset
2.  Train-test split
3.  Scaling
4.  Model training
5.  Evaluation
6.  Visualization

## 🧩 Code Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

data = load_breast_cancer(return_X_y=False, as_frame=True)
dt = data.frame
X, y = load_breast_cancer(return_X_y=True, as_frame=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train_scale, y_train)

y_pred = tree.predict(X_test_scale)

acc = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(f"The Classification Report is: \n{cr}")

cm = confusion_matrix(y_test, y_pred)
print(f"The Confusion Matrix is:\n{cm}")

sns.countplot(x=y)
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='rainbow')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## 📈 Results

- High accuracy
- Class distribution plot
- Confusion matrix heatmap
- Full classification metrics

## 🔧 Installation

```bash
git clone https://github.com/itspriyambhattacharya/Breast-Cancer-ML-Classification-Project.git
pip install -r requirements.txt
classification_breast_cancer.ipynb

```

## 👨‍💻 Author

**Priyam Bhattacharya**\
M.Sc. Computer Science, University of Calcutta

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!
