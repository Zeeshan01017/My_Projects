from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle

df = pd.read_csv("Machine_Learning_Projects/sms_spam_classification/spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","text"]
df["label"] = df["label"].map({"ham":0,"spam":1})

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["text"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

pickle.dump(tfidf, open("Machine_Learning_Projects/sms_spam_classification/vectorizer.pkl","wb"))
pickle.dump(model, open("Machine_Learning_Projects/sms_spam_classification/vectorizer.pkl","wb"))
