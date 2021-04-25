import json
from jsonmerge import merge

usratings = open("traveler_rating_USonly.json")
worldratings = open("traveler_rating_world.json")
ratings = merge(usratings, worldratings)
with open('traveler_rating_MERGED.json', 'w') as f:
  dictratings = json.load(ratings)
  json.dump(dictratings, f)

ustags = open("tag_clouds_USonly.json")
worldtags = open("tag_clouds_world.json")
tags = merge(ustags, worldtags)
with open('tag_clouds_MERGED.json', 'w') as f:
  dicttags = json.load(tags)
  json.dump(dicttags, f)

usreviewquote = open("review_quote_USonly.json")
worldreviewquote = open("review_quote_world.json")
reviewquote = merge(usreviewquote, worldreviewquote)
with open('review_quote_MERGED.json', 'w') as f:
  dictreviewquote = json.load(reviewquote)
  json.dump(dictreviewquote, f)

usreviewcontent = open("review_content_USonly.json")
worldreviewcontent = open("review_content_world.json")
reviewcontent = merge(usreviewcontent, worldreviewcontent)
with open('review_content_MERGED.json', 'w') as f:
  dictreviewcontent = json.load(reviewcontent)
  json.dump(dictreviewcontent, f)

