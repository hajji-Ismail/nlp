import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# 1. قراءة الملف
labels, tweets = [], []

with open("ex07/resources/tweets_train.txt", "r", encoding="utf-8") as f:
    for line in f:
        lab, txt = line.strip().split(",", 1)
        labels.append(lab)
        tweets.append(txt)

df = pd.DataFrame({"label": labels, "tweet": tweets})

# 2. تحويل labels
df["label"] = df["label"].map({
    "positive": 1,
    "neutral": 0,
    "negative": -1
})

# 3. preprocessing
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    return text

df["tweet"] = df["tweet"].apply(clean)

# 4. Bag of Words
vectorizer = CountVectorizer(max_features=500)
X = vectorizer.fit_transform(df["tweet"])

print(X.shape)
print(X)
# 5. DataFrame
count_vectorized_df = pd.DataFrame.sparse.from_spmatrix(
    X,
    columns=vectorizer.get_feature_names_out()
)

# 6. 4th tweet
print(count_vectorized_df.iloc[:3,400:403].to_markdown())

# 7. top 15 words
print(count_vectorized_df.sum().sort_values(ascending=False).head(15))

# 8. add label
count_vectorized_df["label"] = df["label"]

print(count_vectorized_df.iloc[350:354,499:501].to_markdown())