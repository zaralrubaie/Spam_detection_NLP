# -*- coding: utf-8 -*-
"""nlp-spam-detection-clean.ipynb

**Spam Detection with NLP & ML (Minimal Version)**
"""

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv('/kaggle/input/spam-detection/spam.csv')
df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
print(df.head())
print(f"Null values:\n{df.isnull().sum()}")
print(f"Duplicated rows: {df.duplicated().sum()}")

# -----------------------------
# 3. Text Cleaning
# -----------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s']", '', str(text))  # Remove special chars
    return text.lower()

df['Message'] = df['Message'].apply(clean_text)

# -----------------------------
# 4. Tokenization, Stopwords, Lemmatization
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]       # Lemmatize
    return " ".join(tokens)

df['final_text'] = df['Message'].apply(preprocess_text)

# -----------------------------
# 5. Encode Labels
# -----------------------------
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['spamORham'] = label_encoder.fit_transform(df['spamORham'])
print(df['spamORham'].head())

# -----------------------------
# 6. Train/Test Split
# -----------------------------
X = df['final_text']
y = df['spamORham']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 7. Vectorization
# -----------------------------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 8. Train Multinomial Naive Bayes
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

# -----------------------------
# 9. Results
# -----------------------------
results_df = pd.DataFrame({
    "Message": X_test.values,
    "Actual Label": y_test.values,
    "Predicted Label": y_pred
})
print(results_df.head(10))

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 10. Save Model
# -----------------------------
joblib.dump(model, "spam_detector_model.pkl")

# -----------------------------
# 11. Predict Function
# -----------------------------
def predict_message(message):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    return model.predict(vectorized)[0]
