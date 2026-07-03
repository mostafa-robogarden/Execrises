import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = fetch_openml(data_id=31, as_frame=True)
df = data.frame
df.dropna(inplace=True)
numeric_features = ["duration", "credit_amount", "age", "installment_commitment"]
categorical_features = ["checking_status", "credit_history", "purpose", "savings_status"]
target = "class"
X = df[numeric_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_features),("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),])
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
svm = SVC(kernel="rbf", C=1.0, random_state=42) 
svm.fit(X_train_proc, y_train)
y_pred_svm = svm.predict(X_test_proc)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_cm  = confusion_matrix(y_test, y_pred_svm)
print(f"SVM accuracy: {svm_acc:.4f}")
print("SVM confusion matrix (true ↓, pred →):")
print(svm_cm)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_proc, y_train)
y_pred_dt = dt.predict(X_test_proc)
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_cm  = confusion_matrix(y_test, y_pred_dt)
print(f"\nDecision Tree accuracy: {dt_acc:.4f}")
print("Decision Tree confusion matrix (true ↓, pred →):")
print(dt_cm)