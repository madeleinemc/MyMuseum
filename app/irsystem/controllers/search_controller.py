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
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

assert sys.version_info.major == 3


project_name = "Curator"
net_id = "Madeleine Chang (mmc337), Shefali Janorkar (skj28), Esther Lee (esl86), Yvette Hung (yh387), Tiffany Zhong (tz279)"


def tokenize(text):
    temp = re.split('[^a-z]', text.lower())
    words = []
    for w in temp:
        if w != "":
            words.append(w)
    return words


def ltokenize(loc):
    temp = re.split('[^a-z0-9-]', loc.lower())
    nums = []
    for w in temp:
        if w != "":
            nums.append(int(w))
    return nums

# load traveler ratings


museums = []

file = open("museums_file.json")
loaded = json.load(file)
museum_info = {}
for m in loaded:
    museums.append(m)
    museum_info[m] = {}
    for k in loaded[m]:
        museum_info[m][k] = loaded[m][k]

file = open("review_quote_MERGED.json")
# loaded = json.load(file)
raw_review_quotes = json.load(file)
# raw_review_quotes = {}
# for m in loaded:
# 	museums.append(m)
# 	museum_info[m] = {}
# 	for k in loaded[m]:
# 		museum_info[m][k] = loaded[m][k]

# create a TFIDF matrix


def already_tok(d):
    return d


# key = museum, value = index
museum_to_index = {}

i = 0
for m in museums:
    museum_to_index[m] = i
    i += 1

# key = index, value = museum
index_to_museum = {v: k for k, v in museum_to_index.items()}

# get cosine similarity


def get_cos_sim(mus1, mus2, input_doc_mat,
                museum_to_index=museum_to_index):

    v1 = input_doc_mat[museum_to_index[mus1]]
    v2 = input_doc_mat[museum_to_index[mus2]]
    vec1 = np.array(v1)
    vec2 = np.array(v2)

    normvec1 = np.linalg.norm(vec1)
    normvec2 = np.linalg.norm(vec2)

    n = np.dot(vec1, vec2)
    m = np.dot(normvec1, normvec2)

    return n/(m+1)

# construct cosine similarity matrix


def build_museum_sims_cos(num_museums, input_doc_mat, index_to_museum=index_to_museum, museum_to_index=museum_to_index, input_get_sim_method=get_cos_sim):
    cosmat = np.zeros((len(input_doc_mat), len(input_doc_mat)))
    for i in range(len(input_doc_mat)):
        for j in range(len(input_doc_mat)):
            if (i == j):
                cosmat[i][j] = 1.0
            elif (i <= j):
                mus1 = index_to_museum[i]
                mus2 = index_to_museum[j]
                cosmat[i][j] = input_get_sim_method(
                    mus1, mus2, input_doc_mat, museum_to_index)
                cosmat[j][i] = input_get_sim_method(
                    mus1, mus2, input_doc_mat, museum_to_index)
    return cosmat


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    loc = request.args.get('location')
    if not query:
        data = []
        output_message = ''
        # use to make input sticky
        query = ''
    else:
        startsec = time.time()

        tok_query = tokenize(query)

        tok_loc = ltokenize(loc)
        if len(tok_loc) != 2:
            tok_loc = [40, -70]

        l = len(museum_info)
        museum_info[query] = {'ratings': [1, 1, 1, 1, 1], 'tags': tok_query, 'tokenized tags': tok_query,
                              'review titles': tok_query, 'review content': tok_query, 'tokenized content': tok_query, 'location': tok_loc}
        museums.append(query)
        museum_to_index[query] = l
        index_to_museum[l] = query
        print(l)

        # min df originally 10
        tfidf_vec = TfidfVectorizer(min_df=1, max_df=0.8, max_features=5000, analyzer="word",
                                    tokenizer=already_tok, preprocessor=already_tok, token_pattern=None)

        # V rough algo
        # Find the top museums based on tags and reviews
        # find cosine similarity matrices: one for tags and one for reviews (TEMP, may switch to rocchio)
        # multiply 2 cosine similarity matrices: one for tokenized tags and reviews (what about review title?)
        # this code block takes quite a while to run, optimize it?

        # tf-idf matrices
        tfidf_mat_tags = tfidf_vec.fit_transform(
            museum_info[m]['tokenized tags'] for m in museums).toarray()
        tfidf_mat_reviews = tfidf_vec.fit_transform(
            museum_info[m]['tokenized content'] for m in museums).toarray()

        # cosine matrices

        def get_query_cos(num_museums, input_doc_mat, index_to_museum=index_to_museum, museum_to_index=museum_to_index, input_get_sim_method=get_cos_sim):
            qcosmat = np.zeros(len(input_doc_mat))
            mus1 = query
            j = museum_to_index[query]
            for i in range(len(input_doc_mat)):
                mus2 = index_to_museum[i]
                if (i == j):
                    qcosmat[i] = 1.0
                else:
                    qcosmat[i] = input_get_sim_method(
                        mus1, mus2, input_doc_mat, museum_to_index)
            return qcosmat

        def location_mat(num_museums, index_to_museum=index_to_museum, museum_to_index=museum_to_index, input_get_sim_method=get_cos_sim):
            lmat = np.zeros(num_museums)
            mus1 = query
            j = museum_to_index[query]
            for i in range(num_museums):
                mus2 = index_to_museum[i]
                if (i == j):
                    lmat[i] = 90
                else:
                    try:
                        lat = int(museum_info[mus2]['location'][0])
                        long = int(museum_info[mus2]['location'][1])
                    except (ValueError, TypeError):
                        lat = 40
                        long = -70
                    lmat[i] = 90 - ((tok_loc[0] - lat)**2 +
                                    (tok_loc[1] - long)**2)**(1/2)
            return lmat

        # should I add 1 to these matrices?
        num_museums = len(museums)
        #tags_cosine = build_museum_sims_cos(num_museums, tfidf_mat_tags)
        #reviews_cosine = build_museum_sims_cos(num_museums, tfidf_mat_reviews)
        tags_cosine = get_query_cos(num_museums, tfidf_mat_tags)
        reviews_cosine = get_query_cos(num_museums, tfidf_mat_reviews)
        location_matrix = location_mat(num_museums)

        # higher = similar
        # tags and reviews weighted equally here, but can be changed
        multiplied = np.multiply(tags_cosine, reviews_cosine)
        multiplied = np.multiply(multiplied, location_matrix * 0.4)

        # find top n museums, returns dict with format {museum_name: score}

        def get_top_n(museum, n, cosine_mat):
            museum_index = museum_to_index[museum]
            # get index for top n museums, excluding the query "museum"
            top_n_ind = np.argsort(-cosine_mat)[1:n+1]
            top_n_scores = {}

            for t in top_n_ind:
                top_n_scores[index_to_museum[t]] = cosine_mat[t]

            # print(top_n_scores)
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
        #top_5_museums = []
        # for i in top_5:
        #	top_5_museums.append(index_to_museum[i])

        # data is a dict with format {museum_name: {description: x, score: x}}
        data = {}
        for museum in top_5:
            if top_5[museum] != 0:
                data[museum] = {"description": museum_info[museum]
                                ['description'], "score": round(top_5[museum], 2)}

        # clean dataset
        del museums[-1]
        del museum_info[query]
        del museum_to_index[query]
        del index_to_museum[l]

        endsec = time.time()
        mytimediff = endsec - startsec
        strtime = str(mytimediff)[:4]

        # determine output message
        if (len(data) == 0):
            data["    "] = ""
            output_message = "Sorry, there are no matches at this time. Try searching for something else!"
        else:
            output_message = "Your search: " + \
                query + " [" + strtime + " seconds]"

            for name in data:
                # add raw review quotes to data
                data[name]["review_quotes"] = []
                for quote in raw_review_quotes[name]:
                    data[name]["review_quotes"].append(quote)

                # add location info to data
                data[name]["location"] = "(" + str(loaded[name]["location"][0]) + \
                    ", " + str(loaded[name]["location"][1]) + ")"
                data[name]["location_link"] = "https://www.google.com/maps/embed/v1/place?key=AIzaSyD1Bq3RwUmv7r8VG-3p1OWQVGMypRfTv1I&q=" + data[name]["location"]

    # data is in the format of dictionary where key = name of the museum, value = dictionary of information
    # for example,
    # {'museum name1': {"description": "good museum", "score": 0, "review_quotes": [], "location": (123, 123), "location_link": "curator.com"}}

    return render_template('search.html', name=project_name, netid=net_id, search_terms=query, output_message=output_message, data=data)
