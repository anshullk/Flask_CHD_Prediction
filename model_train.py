# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# 1) Load CSV
df = pd.read_csv("./data_cardiovascular_risk.csv")  # change filename

# 2) Basic cleanup: example dataset specifics
# Replace empty strings with NaN
df = df.replace(r'^\s*$', np.nan, regex=True)

# Convert YES/NO to 1/0
df['is_smoking'] = df['is_smoking'].map({'YES':1, 'NO':0})
# sex -> 1 for M, 0 for F
df['sex'] = df['sex'].map({'M':1, 'F':0})

# Ensure numeric cols
numeric_cols = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Target
y = df['TenYearCHD'].astype(int)

# Choose features (drop id and target)
X = df.drop(columns=['id','TenYearCHD'])

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4) Preprocessing pipeline
num_features = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
cat_features = ['education','sex','is_smoking','BPMeds','prevalentStroke','prevalentHyp','diabetes']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# 5) Classifier pipeline (pick RandomForest or LogisticRegression)
clf = Pipeline(steps=[
    ('preproc', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    #('model', LogisticRegression(max_iter=1000))
])

# 6) Train
clf.fit(X_train, y_train)

# 7) Evaluate (quick)
from sklearn.metrics import classification_report, roc_auc_score
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# 8) Save model
joblib.dump(clf, "cvd_model.joblib")
print("Saved model -> cvd_model.joblib")
