import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time

start_time = time.time()

data = pd.read_csv('MBTI 500.csv')

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

X_train, X_test, y_train, y_test = train_test_split(data['posts'], data['type'], test_size=0.2, random_state=42)

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

print(classification_report(y_test, predictions))

end_time = time.time()
elapsed_time = end_time - start_time
print("runtime: " + elapsed_time + " Seconds")