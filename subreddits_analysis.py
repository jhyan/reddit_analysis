'''
Compares relations between different subreddits corpus
and visualizate the result with a heatmap.

Workflow: db data --> panda frame --> token, stem add synonym --> 
tfidf vector --> fit transform docs --> seaborn module draw figure --> snakeviz see profile
'''


from db_connect import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import wordnet
import pandas as pd
import re
import numpy as np
import nltk
import matplotlib
matplotlib.use('agg') # for solving matplotpib compatibility issue
import matplotlib.pyplot as plt
import time


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
# chosen_subreddits = [ 'Fitness', 'java', 'python','datascience', 'MachineLearning']
chosen_subreddits = ['StrangerThings', 'TheUpsideDown', 'Stranger_Things', 'netflix']
subreddit_to_comment_tokens_dict = {} 
for i, subreddit_id in enumerate(all_subreddits):
    sql_query = "SELECT content FROM main_comments WHERE subreddit_id = '%s'" % subreddit_id
    data_local = pd.read_sql(sql_query, con)['content'].tolist()
    subreddit_to_comment_tokens_dict[chosen_subreddits[i]] = data_local


# stemmer and filter stop words, build vocabulary
lancaster = nltk.LancasterStemmer() # apply lancaster stemmer
stopwords = stopwords.words('english') # 153 stop words. a list.
for subreddit, subreddit_comments in subreddit_to_comment_tokens_dict.items():
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
	subreddit_to_comment_tokens_dict[subreddit] = " ".join(tokens_stemed)

################################################## tf-idf heatmap analysis ##############################################
#  transform the documents into tf-idf vectors, then compute the cosine similarity between them
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# TfidfVectorizer is a CountVectorizer followed by TfidfTransformer.
vect = TfidfVectorizer(min_df=1) # <class 'sklearn.feature_extraction.text.TfidfVectorizer'>.  

# chosen subreddits
docs = [subreddit_to_comment_tokens_dict[subreddit] for subreddit in chosen_subreddits]
tfidf_matrix = vect.fit_transform(docs) # <class 'scipy.sparse.csr.csr_matrix'>
# print tfidf_matrix.shape to see the shape of the matrix

print "chosen subreddits: ", chosen_subreddits
x = (tfidf_matrix * tfidf_matrix.T).A # numpy.darray. class object multiplication. implies the cos similarity: one doc tfidf vector cosine another one
df = pd.DataFrame(x, columns=chosen_subreddits, index=chosen_subreddits)

# draw figure
import seaborn as sns
sns.heatmap(df, annot = True, cmap='PuBu') # cmap means color mapping
print df
plt.xticks(rotation = 5)
print 'Runtime: ' + str(int(time.time() - old_time)) + "s"
plt.savefig('./subreddits_analysis_result.png')

