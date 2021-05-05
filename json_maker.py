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

rating_file = open("traveler_rating_MERGED.json")
rating_loaded = json.load(rating_file)
ratings = {}
for r in rating_loaded:
	ratings[r] = []
	for i in rating_loaded[r][::-1]:
		s = i.replace(',', '')
		ratings[r].append(int(s))

csv_file = open("tripadvisor_merged.json")
csv_file_loaded = json.load(csv_file)
csv_content = {}
for r in csv_file_loaded:
	m = r["MuseumName"]
	csv_content[m] = {}
	for s in r:
		csv_content[m][s] = r[s]

museums = list(set(list(csv_content.keys()) + list(ratings.keys())))
inv_museums = {m:v for (v,m) in enumerate(museums)}

# key = museum, value = index
museum_to_index = {}

i = 0
for m in museums:
	museum_to_index[m] = i
	i+=1

# key = index, value = museum
index_to_museum = {v:k for k,v in museum_to_index.items()}

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

review_quote_file = open("review_quote_MERGED.json")
review_quote_file_loaded = json.load(review_quote_file)
review_titles = {}
for r in review_quote_file_loaded:
	review_titles[r] = []
	for i in review_quote_file_loaded[r]:
		s = i.replace(',', '')
		s=s.lower()
		s = re.sub(r'[^\w\s]', '', s)
		review_titles[r].append(s)

review_content_file = open("review_content_MERGED.json")
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
	if (m in tags):
		for t in tags[m]:
			for i in tokenize(t):
					if i not in all_stopwords:
							tok_tags[m].append(i)

# tokenize review content
tok_review = {}
for m in museums:
	tok_review[m] = []
	if (m in review_content):
		for t in review_content[m]:
			for i in tokenize(t):
				if i not in all_stopwords:
					tok_review[m].append(i)


location = {}
for m in museums:
	if (m in csv_content):
		location[m] = (csv_content[m]["Latitude"], csv_content[m]["Longitude"])
	else:
		location[m] = (None, None)
	
description = {}
for m in museums:
	if (m in csv_content):
		description[m] = csv_content[m]["Description"]
	else:
		description[m] = None

free = {}
for m in museums:
	if (m in csv_content):
		free[m] = csv_content[m]["Fee"]
	else:
		free[m] = None


#create dict with all info for each museum
museum_info = {}
for m in museums:
	if not(m in ratings):
		ratings[m] = None
	if not(m in tags):
		tags[m] = None
	if not(m in review_titles):
		review_titles[m] = None
	if not(m in review_content):
		review_content[m] = None
	museum_info[m] = {'description':description[m], 'ratings': ratings[m], 'tags': tags[m], 'tokenized tags': tok_tags[m], 'review titles': review_titles[m], 'review content': review_content[m], 'tokenized content': tok_review[m], 'location': location[m],"free": free[m]}


with open('museums_file.json', 'w') as f:
  json.dump(museum_info, f)

