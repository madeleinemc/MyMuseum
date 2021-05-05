from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
%matplotlib inline
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

import sys
assert sys.version_info.major == 3
from nltk.corpus import stopwords

np.random.seed(42)
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector

import pickle
import heapq
import seaborn as sns
import functools


model = load_model('keras_model.h5')
history = pickle.load(open("history.p", "rb"))

def tokenize(text):
  temp = re.split('[^a-z]', text.lower())
  words = []
  for w in temp:
    if w != "": words.append(w)
  return words

#load traveler ratings
rating_file = open("traveler_rating_MERGED.json")
rating_loaded = json.load(rating_file)
ratings = {}
for r in rating_loaded:
  ratings[r] = []
  for i in rating_loaded[r][::-1]:
    s = i.replace(',', '')
    ratings[r].append(int(s))

#create list of museums and inverted
museums = list(ratings.keys())
inv_museums = {m:v for (v,m) in enumerate(museums)}

#print(museums[0])
#print(inv_museums["Chicago Children's Museum"])

#load tags
tag_clouds_file = open("tag_clouds_MERGED.json")
tag_clouds_file_loaded = json.load(tag_clouds_file)
tags = {}
for r in tag_clouds_file_loaded:
  tags[r] = []
  for i in tag_clouds_file_loaded[r]:
    s = i.replace(',', '')
    s=s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    tags[r].append(s)

all_stopwords = stopwords.words('english')

tok_tags = {}
for m in museums:
  tok_tags[m] = []
  for t in tags[m]:
    for i in tokenize(t):
        if i not in all_stopwords:
            tok_tags[m].append(i)

tokenlist = []
for key in tok_tags.keys():
  tokenlist.append(tok_tags[key]) # try just tags since training takes too long

tokens = []
for museum in tokenlist:
  tokens+= museum

vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1

inv_vocab = {v: k for k, v in vocab.items()}

def prepare_input(text):
    tok_text= tokenize(text)
    x = np.zeros((1, 1, n))
    for word in tok_text:
      if word in vocab.keys():
        x[0][0][vocab[word]] += 1

    return x

# query is what the user has directly typed in
query = ""
seq = query.lower()
seq = prepare_input(seq)
pred = model.predict(seq)
max = np.argmax(pred[0])
pred_word = inv_vocab[max]
if (np.sum(query) == 0): pred_word = "museum" # if there are no key words in the query
print("SUGGESTION:" + pred_word)
