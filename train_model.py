import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

print("🔄 Loading dataset...")

# Load Kaggle dataset
with open("train.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(f"📦 Total recipes: {len(df)}")

# Convert ingredient lists to a single string
df["ingredients_str"] = df["ingredients"].apply(lambda x: " ".join(x))

print("💡 Example recipe:")
print(df[["cuisine", "ingredients_str"]].head())

print("\n🔧 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df["ingredients_str"], df["cuisine"], test_size=0.2, random_state=42
)

# Create a training pipeline
print("📚 Training model...")
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 Model accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, "cuisine_classifier.pkl")
print("💾 Model saved as: cuisine_classifier.pkl")
