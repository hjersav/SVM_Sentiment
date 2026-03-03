import pandas
import pandas as pd
import numpy as np
import sklearn as sk
import nltk as nl
import os
import joblib

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return nl.stem.wordnet.wn.ADJ
    elif tag.startswith('V'):
        return nl.stem.wordnet.wn.VERB
    elif tag.startswith('N'):
        return nl.stem.wordnet.wn.NOUN
    elif tag.startswith('R'):
        return nl.stem.wordnet.wn.ADV
    else:
        return nl.stem.wordnet.wn.NOUN

home = os.getcwd()

np.random.seed(1000)
corpus = pd.read_csv(f"{home}/corpus.csv", encoding="latin-1")

# Tidying up the data
corpus['text'].dropna(inplace=True)
corpus['text'] = [rep.lower() for rep in corpus['text']]
corpus['text'] = [nl.tokenize.word_tokenize(tok) for tok in corpus['text']]

for index, entry in enumerate(corpus['text']):
    Final_words = []
    word_lemmatized = nl.stem.WordNetLemmatizer()
    for word, tag in nl.pos_tag(entry):
        if word not in nl.corpus.stopwords.words('english') and word.isalpha():
            word_Final = word_lemmatized.lemmatize(word, get_wordnet_pos(tag))
            Final_words.append(word_Final)
    corpus.loc[index, 'text_final'] = str(Final_words)
print(corpus['text_final'])
train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(corpus['text_final'], corpus['label'],
                                                                       test_size=0.3)

# Convert __label__1 to 0, __label__2 to 1
encoder = sk.preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

tfidf = sk.feature_extraction.text.TfidfVectorizer(max_features=4000)
tfidf.fit(corpus['text_final'])
train_x_tfidf = tfidf.transform(train_x)
test_x_tfidf = tfidf.transform(test_x)

import time
tn = time.time()
svm_model = sk.svm.LinearSVC()
svm_model.fit(train_x_tfidf, train_y)
prediction = svm_model.predict(test_x_tfidf)
print("SVM Linear ->", sk.metrics.accuracy_score(prediction, test_y)*100)
print("Time Linear ->", time.time()-tn)
