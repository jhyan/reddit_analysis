'''
Compares relations between different subreddits corpus
and visualizate the result with a heatmap.

Workflow: db data --> panda frame --> token, stem add synonym --> 
tfidf vector --> fit transform docs --> seaborn module draw figure --> snakeviz see profile
'''


from db_connect import *
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import wordnet
import pandas as pd
import re
import numpy as np
import nltk
import matplotlib.pyplot as plt
import time

# experiment hyper parameters: vocabulary count, min_df, max_df




VOCAB_CNT = 5
old_time = time.time()

# helper functions
def alpha_filter(w):
    """
    pattern to match a word of alphabetical characters.
    Can be adjusted to be tolerant to several other characters like ! and digits
    """
    pattern = re.compile('^[!_A-Za-z]+$') # the ! character is meaningful
    if pattern.match(w):
        return True
    else:
        return False

def gen_vocabulary(lst, cnt):
	"""
	input a list of tokens, cnt is the number words we need 
	"""
	c = Counter(lst)
	dist = c.most_common(cnt) # distribution
	return [e[0] for e in dist]

# interaction between panda and psql to retrieve python format data
all_subreddits 		= pd.read_sql("SELECT id FROM main_subreddits", con)['id'].tolist()
# all_subreddit_names = pd.read_sql("SELECT name FROM main_subreddits", con)['name'].tolist()

# chosen_subreddits = [ 'Fitness', 'java', 'python','datascience', 'MachineLearning']
chosen_subreddits = ['StrangerThings', 'TheUpsideDown', 'Stranger_Things', 'netflix']
data = {} 
for i, subreddit_id in enumerate(all_subreddits):
    sql_query = "SELECT content FROM main_comments WHERE subreddit_id = '%s'" % subreddit_id
    data_local = pd.read_sql(sql_query, con)['content'].tolist()
    data[chosen_subreddits[i]] = data_local

chosen_subreddits.append("newdoc")
data['newdoc'] = "I love python. Last week I chat with a google engineer about tensorflow. Programming language algorithm".split() # need to be a list of strings to get value



# stemmer and filter stop words, build vocabulary
lancaster = nltk.LancasterStemmer() # apply lancaster stemmer
stopwords = stopwords.words('english') # 153 stop words. a list.
vocab = [] # build the vocabulary for tfidf analysis
for subreddit, subreddit_comments in data.items():
	tokens_stemed = []
	for comment in subreddit_comments:
		tokens = WordPunctTokenizer().tokenize(comment)
		for i in range(len(tokens)):
			try:
				awords = wordnet.synsets(tokens[i])[0].lemma_names() # https://www.cnblogs.com/webRobot/p/6094311.html
				tokens.extend(awords) # similar words list
			except:
				continue
		tokens_stemed.extend([lancaster.stem(t) for t in tokens \
								if alpha_filter(t) and t.lower() not in stopwords]) 
		# vocab.extend(gen_vocabulary(tokens_stemed, VOCAB_CNT))
	# print "{0} has {1} tokens".format(subreddit, len(tokens_stemed))
	data[subreddit] = " ".join(tokens_stemed)
vocab = set(vocab)
# print data["newdoc"]

################################################## tf-idf try out ##############################################
#  transform the documents into tf-idf vectors, then compute the cosine similarity between them
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# np.set_printoptions(formatter={"float":"{:10.3f}".format})
# TfidfVectorizer is a CountVectorizer followed by TfidfTransformer.
vect = TfidfVectorizer(min_df=1)
# vect is <class 'sklearn.feature_extraction.text.TfidfVectorizer'>. The vocab slows the process. What is excactly a vectorizer???
# feature_names = vect.get_feature_names()
# print feature_names[:] # first batch is a few some magic class method names

# chosen subreddits
docs = [data[k] for k in chosen_subreddits] # list of strings
tfidf_matrix = vect.fit_transform(docs) # <class 'scipy.sparse.csr.csr_matrix'>
# print tfidf_matrix.shape

print "chosen subreddits: ", chosen_subreddits
x = (tfidf_matrix * tfidf_matrix.T).A # numpy.darray. class object multiplication. implies the cos similarity: one doc tfidf vector cosine another one
df = pd.DataFrame(x, columns=chosen_subreddits, index=chosen_subreddits)

# draw figure
import seaborn as sns
sns.heatmap(df, annot = True, cmap='PuBu') # cmap is color mapping
plt.xticks(rotation = 0)
print 'Runtime: ' + str(int(time.time() - old_time)) + "s"
plt.show()

