import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.plotting import plot_decision_regions

#st.image("hyperparameter_.jpg")
st.title("Dynamic Dataset and Hyperparameter Tuning for Logistic Regression & SVM")

st.sidebar.header("Customize the Dataset")

# User input for dataset
n_samples = st.sidebar.slider("Number of samples", 100, 5000, 1000)
n_classes = st.sidebar.slider("Number of classes", 2, 5, 3)
n_features = 2
n_informative = 2
n_redundant = 0
random_state = st.sidebar.slider("Random State", 0, 100, 42)

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,  
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_clusters_per_class=1,
    n_classes=n_classes,
    random_state=random_state

)

# DataFrame
df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(n_features)])
df['Target'] = y
st.write("### Generated Dataset (First 5 rows)")
st.write(df.head())

# Select Algorithm
algorithm = st.sidebar.selectbox("Select Algorithm", ("Logistic Regression", "SVM"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

if algorithm == "Logistic Regression":
    # Logistic Regression hyperparameters
    C_value = st.sidebar.slider("Inverse of regularization strength (C)", 0.01, 10.0, 1.0)
    solver = st.sidebar.selectbox("Solver", ("liblinear", "newton-cg", "lbfgs", "sag", "saga"))
    penalty = st.sidebar.selectbox("Penalty", ("l1", "l2", "elasticnet", "none"))
    tol = st.sidebar.slider("Tolerance for stopping criteria (tol)", 0.001, 0.01, 0.001)
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200)
    l1_ratio = st.sidebar.slider("L1 Ratio (only for elasticnet)", 0.0, 1.0, 0.5) if penalty == "elasticnet" else None
    class_weight = st.sidebar.selectbox("Class Weight", (None, "balanced"))

    
    model = LogisticRegression(
        C=C_value,
        solver=solver,
        penalty=penalty if solver != 'newton-cg' else 'l2',
        tol=tol,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        class_weight=class_weight
    )

elif algorithm == "SVM":
    # SVM hyperparameters
    C_value = st.sidebar.slider("C value", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"))
    degree = st.sidebar.slider("Degree (for poly kernel)", 1, 5, 3) if kernel == "poly" else 3
    tol = st.sidebar.slider("Tolerance for stopping criteria (tol)", 0.001, 0.01, 0.001)
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, -1)

    model = SVC(C=C_value, kernel=kernel, degree=degree, tol=tol, max_iter=max_iter)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

st.write("### Classification Metrics")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# Plot decision surface
if n_features == 2:
    st.write("### Decision Surface")
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_train, y_train, clf=model, legend=2)
    plt.title(f"Decision Surface for {algorithm}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(plt)
else:
    st.write("Decision surface can only be displayed when there are exactly 2 features.")