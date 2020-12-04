# Importing the libraries 
import pandas as pd

# Reading the CSV file and labeling the columns
df = pd.read_csv("SMSSpamCollection", sep = "\t", names=["label", "message"])

y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Using stemmer
stemer = PorterStemmer()
corpus_stemer = []

for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemer.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_stemer.append(review)

from nltk.stem import WordNetLemmatizer

# Using the lemmitization
lemmatizer = WordNetLemmatizer()
corpus_lemmatizer = []

for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_lemmatizer.append(review)
  
   
# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer_Bow = CountVectorizer(max_features=5000)
X_Bow_stemmer = count_vectorizer_Bow.fit_transform(corpus_stemer).toarray()
X_Bow_lemmatizer = count_vectorizer_Bow.fit_transform(corpus_lemmatizer).toarray()

from sklearn.model_selection import train_test_split
# BOW with stemmer
X_Bow_stemmer_train, X_Bow_stemmer_test, y_train, y_test = train_test_split(X_Bow_stemmer, y, test_size = 0.20, random_state = 0)
# BOW with lemmatizer
X_Bow_lemmatizer_train, X_Bow_lemmatizer_test, y_train, y_test = train_test_split(X_Bow_lemmatizer, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
# BOW with stemmer and NB
model_bow_stemmer = MultinomialNB().fit(X_Bow_stemmer_train, y_train)
pred_bow_stemmer = model_bow_stemmer.predict(X_Bow_stemmer_test)
# BOW with lemmatizer and NB
model_bow_lemmatizer = MultinomialNB().fit(X_Bow_lemmatizer_train, y_train)
pred_bow_lemmatizer = model_bow_lemmatizer.predict(X_Bow_lemmatizer_test)

from sklearn.metrics import accuracy_score
accuracy_score_Bow_stemmer = accuracy_score(y_test, pred_bow_stemmer)
accuracy_score_Bow_lemmatizer = accuracy_score(y_test, pred_bow_lemmatizer)

# Everythig same using TFIDFs
from sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer_tfidf = TfidfVectorizer(max_features=5000)
X_tfidf_stemmer = count_vectorizer_tfidf.fit_transform(corpus_stemer).toarray()
X_tfidf_lemmatizer = count_vectorizer_tfidf.fit_transform(corpus_lemmatizer).toarray()

X_tfidf_stemmer_train, X_tfidf_stemmer_test, y_train, y_test = train_test_split(X_tfidf_stemmer, y, test_size = 0.20, random_state = 0)
X_tfidf_lemmatizer_train, X_tfidf_lemmatizer_test, y_train, y_test = train_test_split(X_tfidf_lemmatizer, y, test_size = 0.20, random_state = 0)

model_tfidf_stemmer = MultinomialNB().fit(X_tfidf_stemmer_train, y_train)
pred_tfidf_stemmer = model_tfidf_stemmer.predict(X_tfidf_stemmer_test)
model_tfidf_lemmatizer = MultinomialNB().fit(X_tfidf_lemmatizer_train, y_train)
pred_tfidf_lemmatizer = model_tfidf_lemmatizer.predict(X_tfidf_lemmatizer_test)

accuracy_score_tfidf_stemmer = accuracy_score(y_test, pred_tfidf_stemmer)
accuracy_score_tfidf_lemmatizer = accuracy_score(y_test, pred_tfidf_lemmatizer)
