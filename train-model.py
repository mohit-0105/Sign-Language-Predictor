import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---- LOAD DATA ----
CSV_FILE = "hand_landmarks.csv"
df = pd.read_csv(CSV_FILE)

print(f"Total samples: {len(df)}")
print("Samples per sign:")
print(df["label"].value_counts())

# ---- SPLIT FEATURES AND LABELS ----
X = df.drop("label", axis=1)   # all the x,y,z coordinates
y = df["label"]                 # the sign name

# ---- TRAIN/TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- TRAIN RANDOM FOREST ----
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- EVALUATE ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- SAVE THE TRAINED MODEL ----
joblib.dump(model, "sign_model.pkl")
print("\nModel saved as 'sign_model.pkl'")