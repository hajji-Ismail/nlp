import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# =========================
# 1. Load data
# =========================
tweets = []
labels = []

with open("tweets_train.txt", "r", encoding="utf-8") as f:
    for line in f:
        # Adapt this depending on your file format
        # Example assumes: "label \t tweet"
        parts = line.strip().split("\t")
        if len(parts) == 2:
            label, tweet = parts
            tweets.append(tweet)
            labels.append(int(label))

# =========================
# 2. Preprocess tweets
# =========================
tweets_clean = [preprocess(tweet) for tweet in tweets]

# =========================
# 3. CountVectorizer (BoW)
# =========================
vectorizer = CountVectorizer(max_features=500)

X = vectorizer.fit_transform(tweets_clean)

# =========================
# 4. Check shape
# =========================
print("Shape of word count matrix:", X.shape)

# =========================
# 5. Convert to DataFrame
# =========================
feature_names = vectorizer.get_feature_names_out()

count_vectorized_df = pd.DataFrame.sparse.from_spmatrix(
    X,
    columns=feature_names
)

# =========================
# 6. Show 4th tweet counts
# =========================
print("\nToken counts for 4th tweet:")
print(count_vectorized_df.iloc[3])

# =========================
# 7. Top 15 most frequent words
# =========================
word_counts = np.array(X.sum(axis=0)).flatten()

word_freq = pd.DataFrame({
    "word": feature_names,
    "count": word_counts
})

top15 = word_freq.sort_values(by="count", ascending=False).head(15)

print("\nTop 15 most frequent words:")
print(top15)

# =========================
# 8. Add label column
# =========================
count_vectorized_df["label"] = labels

# =========================
# 9. Final preview
# =========================
print("\nFinal DataFrame preview:")
print(count_vectorized_df.head())