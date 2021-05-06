import time
from nltk.corpus import stopwords
import sys
from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# Commented out IPython magic to ensure Python compatibility.
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
# %matplotlib inline
import re
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
#ML
import tensorflow as tf
from keras.models import Sequential, load_model

import pickle
import heapq
import functools

assert sys.version_info.major == 3


project_name = "Curator"
net_id = "Madeleine Chang (mmc337), Shefali Janorkar (skj28), Esther Lee (esl86), Yvette Hung (yh387), Tiffany Zhong (tz279)"

# tokenize functions

def tokenize(text):
	temp = re.split('[^a-z]', text.lower())
	words = []
	for w in temp:
			if w != "":
					words.append(w)
	return words


def ltokenize(loc):
	temp = re.split('[^-?0-9.]', loc.lower())
	nums = []
	for w in temp:
			if w != "":
					nums.append(float(w))
	return nums

# load museums file

museums = []

file = open("museums_file.json")
loaded = json.load(file)
museum_info = {}
for m in loaded:
	museums.append(m)
	museum_info[m] = {}
	for k in loaded[m]:
		museum_info[m][k] = loaded[m][k]

# review quotes to display
file = open("review_quote_MERGED.json")
raw_review_quotes = json.load(file)

def already_tok(d):
	return d

# key = museum, value = index
museum_to_index = {}

# key = index, value = museum
index_to_museum = {}

i = 0
for m in museums:
	museum_to_index[m] = i
	if museum_info[m].get('description') is None:
		museum_info[m]['description'] = m
	index_to_museum[i] = m
	i += 1

# get cosine similarity


def get_cos_sim(mus1, mus2, input_doc_mat, museum_to_index=museum_to_index):

	v1 = input_doc_mat[museum_to_index[mus1]]
	v2 = input_doc_mat[museum_to_index[mus2]]
	vec1 = np.array(v1)
	vec2 = np.array(v2)

	normvec1 = np.linalg.norm(vec1)
	normvec2 = np.linalg.norm(vec2)

	n = np.dot(vec1, vec2)
	m = np.dot(normvec1, normvec2)

	return n/(m+1)

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	loc = request.args.get('location')
	free = request.args.get("freeSearch")
	family = request.args.get("familySearch")
	if not query:
		data = []
		output_message = ''
		# use to make input sticky
		query = ''
	else:

		# ML COMPONENT (also in ML.py) ********************************

		model = load_model('keras_model.h5')
		history = pickle.load(open("history.p", "rb"))

		tokenlist = []
		for m in museum_info:
			tokenlist.append(museum_info[m]['tokenized tags']) # try just tags since training takes too long

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
				x = np.zeros((1, 1, len(vocab)))
				for word in tok_text:
					if word in vocab.keys():
						x[0][0][vocab[word]] += 1

				return x

		# query is what the user has directly typed in
		seq = query.lower()
		seq = prepare_input(seq)
		pred = model.predict(seq)
		max = np.argmax(pred[0])
		pred_word = inv_vocab[max]
		if (len(query) == 0): pred_word = "museum" # if there are no key words in the query

		# END OF ML COMPONENT ********************************

		startsec = time.time()

		tok_query = tokenize(query)

		tok_loc = ltokenize(loc)
		if len(tok_loc) != 2:
			tok_loc = [40.0, -70.0]
			loc = True
		else:
			temp = tok_loc[0]
			tok_loc[0] = tok_loc[1]
			tok_loc[1] = temp

		l = len(museum_info)
		Qkey = "Q:" + query
		museum_info[Qkey] = {'ratings': [1, 1, 1, 1, 1], 'tags': tok_query, 'tokenized tags': tok_query, 'review titles': tok_query, 'review content': tok_query, 'tokenized content': tok_query, 'location': tok_loc, 'description': query}
		museums.append(Qkey)
		museum_to_index[Qkey] = l
		index_to_museum[l] = Qkey

		tok_museums = {}
		tok_description = {}
		for m in museums:
			tok_museums[m] = tokenize(m)
			tok_description[m] = tokenize(museum_info[m]['description'])

		distances = {}
		distances[Qkey] = 0.0

		# min df originally 10
		tfidf_vec = TfidfVectorizer(min_df=1, max_df=0.8, max_features=5000, analyzer="word", tokenizer=already_tok, preprocessor=already_tok, token_pattern=None)

		# V rough algo
		# Find the top museums based on tags and reviews
		# find cosine similarity matrices: one for tags and one for reviews (TEMP, may switch to rocchio)
		# multiply 2 cosine similarity matrices: one for tokenized tags and reviews (what about review title?)
		# this code block takes quite a while to run, optimize it?

		# tf-idf matrices
		tfidf_mat_tags = tfidf_vec.fit_transform(museum_info[m]['tokenized tags'] for m in museums).toarray()
		tfidf_mat_reviews = tfidf_vec.fit_transform(museum_info[m]['tokenized content'] for m in museums).toarray()
		tfidf_mat_names = tfidf_vec.fit_transform(tok_museums[m] for m in museums).toarray()
		tfidf_mat_description = tfidf_vec.fit_transform(tok_description[m] for m in museums).toarray()

		# cosine matrices
		def get_query_cos(num_museums, input_doc_mat, index_to_museum=index_to_museum, museum_to_index=museum_to_index, input_get_sim_method=get_cos_sim):
			qcosmat = np.zeros(len(input_doc_mat))
			mus1 = Qkey
			j = museum_to_index[Qkey]
			for i in range(len(input_doc_mat)):
				mus2 = index_to_museum[i]
				if (i == j):
					qcosmat[i] = 1.0
				else:
					qcosmat[i] = input_get_sim_method(mus1, mus2, input_doc_mat, museum_to_index)
			return qcosmat

		def location_mat(num_museums, index_to_museum=index_to_museum, museum_to_index=museum_to_index):
				lmat = np.zeros(num_museums)
				mus1 = Qkey
				j = museum_to_index[Qkey]
				for i in range(num_museums):
					mus2 = index_to_museum[i]
					if (i == j):
						lmat[i] = 1.0
					else:
						try:
							lat = float(museum_info[mus2]['location'][0])
							long = float(museum_info[mus2]['location'][1])
						except (ValueError, TypeError):
							lat = 0.0
							long = 0.0
						c1 = (tok_loc[0], tok_loc[1])
						c2 = (lat, long)
						distance = haversine(c1, c2)
						distances[mus2] = distance
						lmat[i] = (40043 - distance)/40043
				return lmat

		# should I add 1 to these matrices?
		num_museums = len(museums)
		tags_cosine = get_query_cos(num_museums, tfidf_mat_tags)
		reviews_cosine = get_query_cos(num_museums, tfidf_mat_reviews)
		names_cosine = get_query_cos(num_museums, tfidf_mat_names)
		description_cosine = get_query_cos(num_museums, tfidf_mat_description)
		location_matrix = location_mat(num_museums)

		# for m in tok_museums:
		# 	if (m == Qkey):
		# 		pass
		# 	elif (query.lower() == m.lower()):
		# 		k = museum_to_index[m]
		# 		tags_cosine[k] = 1.0
		# 		reviews_cosine[k] = 1.0
		# 		names_cosine[k] = 1.0
		# 		description_cosine[k] = 1.0
		# 		location_matrix[k] = 1.0

		# higher = similar
		# tags and reviews weighted equally here, but can be changed
		##multiplied = np.multiply((tags_cosine), (reviews_cosine))
		#print(multiplied[i])
		##multiplied = np.multiply(multiplied, description_cosine)
		#print(multiplied[i])
		#multiplied = np.multiply(multiplied, names_cosine)
		#print(multiplied[i])
		##multiplied = np.multiply(multiplied, location_matrix)
		#multiplied = multiplied + (location_matrix*0000.3)
		#print(multiplied[i])

		multiplied = tags_cosine*0.2 + reviews_cosine*0.1 + description_cosine*0.05 + (location_matrix)*0.65

		# find top n museums, returns dict with format {museum_name: score}
		def getFreeMuseums():
				freeMuseumIndices = []
				for museum in loaded:
						if loaded[museum]["free"] == "Yes":
								freeIndex = museum_to_index[museum]
								freeMuseumIndices.append(freeIndex)
								# print(museum)
								# print(museum)
								# print("freemuseumIndice")
								# print(freeMuseumIndices)
				return freeMuseumIndices

		def getFamilyMuseums():
				familyMuseumIndices = []
				for museum in loaded:
						if loaded[museum]["ratings"] == "Yes":
								familyIndex = museum_to_index[museum]
								familyMuseumIndices.append(familyIndex)
								# print(museum)
								# print(museum)
								# print("freemuseumIndice")
								# print(freeMuseumIndices)
				return familyMuseumIndices

		# find top n museums, returns dict with format {museum_name: score}

		def get_top_n(museum, n, cosine_mat):
				freeMuseumIndices = getFreeMuseums()
				familyMuseumIndices = getFamilyMuseums()
				top_n_scores = {}
				if (free == "on") and (family == "on"):
						top_n_ind = np.argsort(-cosine_mat)[1:]
						top_free_family = []
						for indice in top_n_ind:
								if indice in freeMuseumIndices and indice in familyMuseumIndices:
										top_free_family.append(indice)
						for t in top_free_family[:n+1]:
								top_n_scores[index_to_museum[t]] = cosine_mat[t]
				if (free == "on") and (family != "on"):
						top_n_ind = np.argsort(-cosine_mat)[1:]
						top_free = []
						for indice in top_n_ind:
								if indice in freeMuseumIndices:
										top_free.append(indice)
						for t in top_free[:n+1]:
								top_n_scores[index_to_museum[t]] = cosine_mat[t]
				if (free != "on") and (family == "on"):
						top_n_ind = np.argsort(-cosine_mat)[1:]
						top_family = []
						for indice in top_n_ind:
								if indice in familyMuseumIndices:
										top_family.append(indice)
						for t in top_family[:n+1]:
								top_n_scores[index_to_museum[t]] = cosine_mat[t]
				if (free != "on") and (family != "on"):
						top_n_ind = np.argsort(-cosine_mat)[1:n+1]
						for t in top_n_ind:
								top_n_scores[index_to_museum[t]] = cosine_mat[t]
				return top_n_scores

		top_5 = get_top_n(query, 5, multiplied)

		# TODO
		# 1. If time allows, narrow down location based on latitude and longitude
		# 2. Not include query

		# print("Top 20 Matches for The Mariners' Museum & Park\n")
		# i = 1
		# for t in top_5:
		# 	print(str(i) + ': ' + index_to_museum[t])
		# 	i+=1
		# top_5_museums = []
		# for i in top_5:
		#	top_5_museums.append(index_to_museum[i])

		# data is a dict with format {museum_name: {description: x, score: x}}
		data = {}
		for museum in top_5:
			if top_5[museum] >= 0.65:
				data[museum] = {"description": museum_info[museum]['description'], "score": round(top_5[museum], 3), "distance": round(distances[museum], 3)}

		# clean dataset
		del museums[-1]
		del museum_info[Qkey]
		del museum_to_index[Qkey]
		del index_to_museum[l]

		endsec = time.time()
		mytimediff = endsec - startsec
		strtime = str(mytimediff)[:4]

		# determine output message
		if (len(data) == 0):
			# data["    "] = ""
			output_message = "Sorry, there are no matches at this time. Try searching for something else, perhaps \"" + pred_word + "\""
		else:
			output_message = query + " [" + strtime + " seconds]." + "\n" + "Would you like to search for \"" + pred_word + "\"?"

			for name in data:
				# add raw review quotes to data
				data[name]["review_quotes"] = []
				if raw_review_quotes.get(name) is not None:
					for quote in raw_review_quotes[name]:
						data[name]["review_quotes"].append(quote)

				# add location info to data
				data[name]["location"] = "(" + str(loaded[name]["location"][0]) + ", " + str(loaded[name]["location"][1]) + ")"
				data[name]["location_link"] = "https://www.google.com/maps/embed/v1/place?key=AIzaSyD1Bq3RwUmv7r8VG-3p1OWQVGMypRfTv1I&q=" + data[name]["location"]

	# data is in the format of dictionary where key = name of the museum, value = dictionary of information
	# for example,
	# {'museum name1': {"description": "good museum", "score": 0, "review_quotes": [], "location": (123, 123), "location_link": "curator.com"}}

	return render_template('search.html', name=project_name, netid=net_id, search_terms=query, output_message=output_message, data=data)
