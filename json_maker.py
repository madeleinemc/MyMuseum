import json
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 

def tokenize(text):
  temp = re.split('[^a-z]', text.lower())
  words = []
  for w in temp:
    if w != "": words.append(w)
  return words

rating_file = open("traveler_rating_USonly.json")
rating_loaded = json.load(rating_file)
ratings = {}
for r in rating_loaded:
	ratings[r] = []
	for i in rating_loaded[r][::-1]:
		s = i.replace(',', '')
		ratings[r].append(int(s))

museums = list(ratings.keys())
inv_museums = {m:v for (v,m) in enumerate(museums)}

# key = museum, value = index
museum_to_index = {}

i = 0
for m in museums:
	museum_to_index[m] = i
	i+=1

# key = index, value = museum
index_to_museum = {v:k for k,v in museum_to_index.items()}

tag_clouds_file = open("tag_clouds_USonly.json")
tag_clouds_file_loaded = json.load(tag_clouds_file)
tags = {}
for r in tag_clouds_file_loaded:
	tags[r] = []
	for i in tag_clouds_file_loaded[r]:
		s = i.replace(',', '')
		s=s.lower()
		s = re.sub(r'[^\w\s]', '', s)
		tags[r].append(s)

review_quote_file = open("review_quote_USonly.json")
review_quote_file_loaded = json.load(review_quote_file)
review_titles = {}
for r in review_quote_file_loaded:
	review_titles[r] = []
	for i in review_quote_file_loaded[r]:
		s = i.replace(',', '')
		s=s.lower()
		s = re.sub(r'[^\w\s]', '', s)
		review_titles[r].append(s)

review_content_file = open("review_content_USonly.json")
review_content_file_loaded = json.load(review_content_file)
review_content = {}
for r in review_content_file_loaded:
	review_content[r] = []
	for i in review_content_file_loaded[r]:
		s = i.replace(',', '')
		s=s.lower()
		s=re.sub('\n','',s)
		s=re.sub('\xa0','',s)
		s = re.sub(r'[^\w\s]', '', s)
		review_content[r].append(s)

all_stopwords = stopwords.words('english')

tok_tags = {}
for m in museums:
	tok_tags[m] = []
	for t in tags[m]:
		for i in tokenize(t):
				if i not in all_stopwords:
						tok_tags[m].append(i)

# tokenize review content
tok_review = {}
for m in museums:
	tok_review[m] = []
	for t in review_content[m]:
		for i in tokenize(t):
			if i not in all_stopwords:
				tok_review[m].append(i)

# possibly clean up the tokens

#create dict with all info for each museum
museum_info = {}
for m in museums:
	museum_info[m] = {'ratings': ratings[m], 'tags': tags[m], 'tokenized tags': tok_tags[m], 'review titles': review_titles[m], 'review content': review_content[m], 'tokenized content': tok_review[m]}


with open('museums_file.json', 'w') as f:
  json.dump(museum_info, f)

