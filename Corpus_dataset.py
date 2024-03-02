from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sklearn as sk
import json
import pickle

financial_corpus_df = pd.read_csv('/training_data.csv')
financial_corpus_df

financial_corpus_df['category'].unique()

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(financial_corpus_df['category'])
financial_corpus_df['label'] = label_encoder.transform(financial_corpus_df['category'])

financial_corpus_df['label'].unique()

financial_corpus_df

vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 1000)

x = financial_corpus_df['body']
y = financial_corpus_df['label']

vectorized_x = vectorizer.fit_transform(x)

rf_clf = RandomForestClassifier()

rf_clf.fit(vectorized_x,y)

pickle.dump(rf_clf, open('financial_text_classifier.pkl','wb'))
pickle.dump(vectorizer, open('financial_text_vectorizer.pkl','wb'))
pickle.dump(label_encoder, open('financial_text_encoder.pkl','wb'))
