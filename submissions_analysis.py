'''
A mini search recommendation system for reddit.
Input key words --> output a post most related to the input.

Workflow:
 db data --> panda frame --> gen docs (token, stem add synonym) --> 
 add question to the docs--> get cosine of rows in tfidf matrix --> 
 filter by time/length/score --> save the result to a local file.
'''


from db_connect import *
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from collections import defaultdict
from nltk.corpus import wordnet
from scipy.spatial.distance import cosine
from datetime import datetime
import pandas as pd
import re
import numpy as np
import nltk
import matplotlib.pyplot as plt
import time


# add this line before the running: sudo service postgresql restart
# sudo psql testdb postgres; \d; then select directly
# shortlink. redd.it/id

# VOCAB_CNT = 5
SCORE_CRITERIA = 3
LENGTH_CRITERIA = 2
TIME_CRITERIA = 1
OLD_TIME = time.time()
DEBUG = True
SYNONYMS = True
SYNONYMS_CNT = 3
TAG_AMPLIFIER_FACTOR = 3 # the tag amplification factor
TAG_AMPLIFIER_SET = set(['V','N']) # tags to be amplified
CHOSEN_SUBREDDITS = ['StrangerThings', 'TheUpsideDown', 'Stranger_Things', 'netflix']
QUESTIONS = None
with open('question.txt', 'r') as f:
	QUESTIONS = f.read().split('\n')

# helper functions
def alpha_filter(w):
    '''
    pattern to match a word of alphabetical characters.
    Can be adjusted to be tolerant to several other characters like ! and digits
    '''
    pattern = re.compile('^[!_A-Za-z]+$') # the ! character is meaningful
    if pattern.match(w):
        return True
    else:
        return False

def gen_vocabulary(lst, cnt):
	'''
	input a list of tokens, cnt is the number words we need 
	'''
	c = Counter(lst)
	dist = c.most_common(cnt) # distribution
	return [e[0] for e in dist]

def convert_secs(secs_float):
	'''
	Takes a number of seconds and returns datetime
	'''
	return datetime.fromtimestamp(secs_float).strftime("%B %d, %Y %I:%M:%S")


# analysis functions
def gen_data(all_subreddits):
	'''
	conceive 2-layer dict -->
	{subreddit_id:{submission_id:contents corpus}}
	'''

	data = defaultdict(dict)
	submission_time_lookup = {}
	for i, subreddit_id in enumerate(all_subreddits): 
		# create inner dict
		data_inner = defaultdict(list)
		comments_query = "SELECT content, submission_id FROM main_comments WHERE subreddit_id = '%s'" % subreddit_id
		comments_frame = pd.read_sql(comments_query, con)
		for submission_id, content in zip(comments_frame.submission_id, comments_frame.content):
			data_inner[submission_id].append(content)
		# create outer dict
		data[CHOSEN_SUBREDDITS[i]] = data_inner

		# create lookup table --> submission_id : time
		submission_query = "SELECT id, created FROM main_submissions WHERE subreddit_id = '%s'" % subreddit_id
		submission_frame = pd.read_sql(submission_query, con)
		submission_time_lookup.update(dict(zip(submission_frame.id, submission_frame.created)))
	if DEBUG == 2:
		print ("data keys: ", data.keys())
	return data, submission_time_lookup

def gen_docs(data, docs, submissions, 
	stemmer = nltk.LancasterStemmer(), stopwords = stopwords.words('english')):
	'''
	data: {subreddit_id:{submission_id:contents corpus}}
	docs: gen list of tokens
	submission: [(submission_id, token_length), ... ()] 
	stemmer: default apply lancaster stemmer
	stopwords: default 153 english words
	four layers: subreddit --> submission --> comments --> token
	'''
	if DEBUG == 2:
		print ("submission id and length relations:")
	for subreddit, inner in data.items():
		for submission_id, comments in inner.items():
			tokens_stemed = []
			for comment in comments:
				tokens = WordPunctTokenizer().tokenize(comment)
				if SYNONYMS:
					for i in range(len(tokens)):
						try:
							awords = wordnet.synsets(tokens[i])[0].lemma_names()[:SYNONYMS_CNT] # https://www.cnblogs.com/webRobot/p/6094311.html
							tokens.extend(awords) # similar words list
						except:
							continue
				# amplify the verb
				tokens_tags = nltk.pos_tag(tokens)
				for token, tag in tokens_tags:
					if alpha_filter(token) and token.lower() not in stopwords:
						if tag in TAG_AMPLIFIER_SET:
							tokens_stemed.extend([token] * TAG_AMPLIFIER)
						else:
							tokens_stemed.append(token)

				# tokens_stemed.extend([stemmer.stem(t) for t in tokens 
				# 						if alpha_filter(t) and t.lower() not in stopwords]) 
			# vocab.extend(gen_vocabulary(tokens_stemed, VOCAB_CNT))
			if DEBUG == 2:
				print ("{0} has {1} tokens".format(submission_id, len(tokens_stemed)))
			submissions.append((submission_id, len(tokens_stemed)))
			docs.append(" ".join(tokens_stemed))

def addon_docs(submission_time_lookup, QUESTION):
	'''
	add the question to the docs
	also modify submissions tuple (id, time)
	'''
	add_on = {}
	add_on[0] = {0:[QUESTION]} # need to be a int -> list relation
	submission_time_lookup[0] = int(time.time())
	gen_docs(add_on, docs, submissions)
	# vocab = set(vocab)
	if DEBUG == 2:
		print ('there are {0} submissions and {1} docs\n'.format(len(submissions), len(docs)))

def gen_stats(docs):
	'''
	docs: list of tokens
	stats: a tuple of (submission_id, token_length, tf-idf_score, submission_time)
	'''
	vect = TfidfVectorizer(min_df=1)
	tfidf_matrix = vect.fit_transform(docs) # <class 'scipy.sparse.csr.csr_matrix'>
	row, col = tfidf_matrix.shape # unpack two elements

	stats = []
	for i in range(row-1): # exclude the last element (the actual question) 
		# cosine more smalller, more similar. So use 1-x to reverse the interpretation
		try:
			score = 1 - cosine(tfidf_matrix[i].todense(), tfidf_matrix[row-1].todense())
		except:
			score = 0
		stats.append([submissions[i][0], "%5d" % submissions[i][1], round(score,4),
					 submission_time_lookup[submissions[i][0]]])
					# convert_secs(submission_time_lookup[submissions[i][0]])))
	return stats

def match(stats, QUESTION):
	'''
	select the best fit submission through multiply layers
	according to the score, created time, token length
	'''
	nlp_related = sorted(stats, key=lambda x: -x[2])[:SCORE_CRITERIA]
	print ("|---id---|-length-|--score-|---time-------|--url------------")
	if DEBUG:
		for i in nlp_related:
			print (i + ['url: redd.it/{0}'.format(i[0])])
		print ("+++++finished select nlp score\++++++\n")
	length_related = sorted(nlp_related, key = lambda x: -int(x[1]))[:LENGTH_CRITERIA]
	if DEBUG:
		for i in length_related:
			print (i + ['url: redd.it/{0}'.format(i[0])])
		print ("+++++finished select length+++++\n")
	time_related = sorted(length_related, key = lambda x: -int(x[-1]))[:TIME_CRITERIA]
	if DEBUG:	
		for i in time_related:
			print (i + ['url: redd.it/{0}'.format(i[0])])
		print ("+++++finished select time+++++\n")

	print ("Final match id for question: {0}".format(QUESTION))
	for i in time_related:
		print ("redd.it/{0}".format(i[0]))



if __name__ == '__main__':
	# define chosen subreddits
	
	# interaction between panda and psql
	all_subreddits = pd.read_sql("SELECT id FROM main_subreddits", con)['id'].tolist()
	
	# retrieve python data structure format data
	data, submission_time_lookup = gen_data(all_subreddits) # data: {subreddit_id:{submission_id:contents corpus}}. submission_time_lookup: {id: time}
	submissions, docs = [], [] # submissions: list of tuple (id, token_length). docs: list of tokens
	
	# generate docs & add the question
	gen_docs(data, docs, submissions)

	for i,q in enumerate(QUESTIONS):
		print ('Question {0}: {1}'.format(i+1,q))
		addon_docs(submission_time_lookup, q)
		# genetate stats and output the final match!
		stats = gen_stats(docs)
		match(stats, q)
		docs.pop() # pop the last doc(question) and prepare for the next one 
		print ('Runtime till now: ' + str(int(time.time() - OLD_TIME)) + "s")
		print ('\n\n\n\n')
