import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].apply(transform_text)

# Train model
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["text"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)   # ⭐ THIS LINE TRAINS THE MODEL ⭐

# Save trained model
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("MODEL TRAINED AND SAVED")
