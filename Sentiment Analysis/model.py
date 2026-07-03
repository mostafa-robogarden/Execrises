import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

data_list = []
DIR = 'dataset'
for filename in ['Train.csv', 'Valid.csv', 'Test.csv']:
    df = pd.read_csv(os.path.join(DIR, filename), index_col=None, header=0)
    data_list.append(df)
data = pd.concat(data_list, axis=0, ignore_index=True)
x = data["text"]
y = data["label"]
x_train, x_test, y_train ,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
modal = LogisticRegression()
modal.fit(x_train_vec, y_train)
y_pred = modal.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy", accuracy)
Report = classification_report(y_test, y_pred)
print(Report)

new_review = ["I didn't like the ending but, overall, it was good"]
new_review_vec = vectorizer.transform(new_review)
pred_sentiment = modal.predict(new_review_vec)
if(pred_sentiment == 0):
    print("Review is negative")
else:
    print("Review is positive")