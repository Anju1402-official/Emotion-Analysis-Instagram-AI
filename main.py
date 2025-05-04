import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample Instagram data
data = {
    'text': [
        "I love this!", 
        "You ruined my day", 
        "So beautiful 💕", 
        "I hate this post", 
        "Made me smile"
    ],
    'emotion': ['happy', 'angry', 'happy', 'angry', 'happy']
}
df = pd.DataFrame(data)

# Text cleaning
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

df['cleaned'] = df['text'].apply(clean_text)

# Vectorize
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['emotion']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Report
print(classification_report(y_test, y_pred))
