'''
Scrape the reddit data and save to the database.
'''

import os
import pandas as pd
import praw
import sys
import time
from datetime import datetime
from create_table import *
from multiprocessing.dummy import Pool


HOT_CNT = 100
COMMENTS_THRESHOLD = 500

old_time = time.time()

def convert_secs(secs_float):
	"""
	Takes a number of seconds and returns datetime
	"""
	return datetime.fromtimestamp(secs_float).strftime("%B %d, %Y %I:%M:%S")



reddit = praw.Reddit(client_id='qrbjP0JQu3Uo1Q',
                     client_secret='lyFcQOLLN1OeH1a4-BE5WIJeWgM',
                     user_agent = 'doe nlp usage', # user_agent is just a description
                     username='sisyphus_bot',
                     password='Sympler1~') 
print("I am reddit bot {0}".format(reddit.user.me()))


# chosen_subreddits = [ 'Fitness', 'java', 'python','datascience', 'MachineLearning']
chosen_subreddits = ['StrangerThings', 'TheUpsideDown', 'Stranger_Things', 'netflix']
for i in range(len(chosen_subreddits)):
    subreddit = reddit.subreddit(chosen_subreddits[i])
    db_tuple = (subreddit.id, chosen_subreddits[i], subreddit.title, int(subreddit.created), 0)
    print db_tuple
    sql_query = """INSERT INTO main_subreddits VALUES (%s, %s, %s, %s, %s)"""
    cur.execute(sql_query, db_tuple) # use tuple to insert


############# database is down ##############
# subreddit -> pandas dataframe
pull_df = pd.read_sql("SELECT * FROM main_subreddits", con) # <clddlass 'pandas.core.frame.DataFrame'>

# Parse subreddits-->submissions-->comments and save to SQL
for i in range(pull_df.shape[0]):
    t_start = time.time()
    print ("Subreddit " + str(i + 1) + " / " + 
    	   str(pull_df.shape[0]) + ": " + pull_df['title'][i])
    # deal with subreddit
    subreddit = reddit.subreddit(pull_df['name'][i])
    subreddit_id = pull_df['id'][i]
    submissions_cnt = 0
    comment_count = 0
    # deal with submissions
    for submission in subreddit.hot(limit=HOT_CNT): # this step connects the internet?
        submissions_cnt += 1
        submission.comments.replace_more(limit = 0); # For deep comments
        submission_tuple = (submission.id, subreddit_id, submission.selftext, int(submission.created))
        submission_query =  "INSERT INTO main_submissions VALUES (%s, %s, %s, %s)"
        cur.execute(submission_query, submission_tuple)
        # deal with comments
        comment_tuples = []
        comment_list = submission.comments.list()
        for comment in comment_list:
            comment_tuples.append((comment.id , subreddit_id, 
            					   submission.id, comment.body))
        if len(comment_tuples) > 0: # smart, insert in one sentense
            comment_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x) for x in comment_tuples)
            cur.execute("INSERT INTO main_comments VALUES " + comment_str)

        comment_count += len(comment_list)
        # The rest is for pretty tracking of progress
        sys.stdout.write("\rComments processed: {0}, Submissions processed: {1}".format(comment_count, submissions_cnt)) # print i, comma means wait for all to finish. \r and \n cannot be used together
        sys.stdout.flush()
        if comment_count > COMMENTS_THRESHOLD: break
    
    print '  Runtime: ' + str(int(time.time() - t_start)) + "s"
    print '-' * 50

#Check the total amount of submissions and comments downloaded. It's rolling.
all_comments = pd.read_sql("SELECT * FROM main_comments", con)
all_submissions = pd.read_sql("SELECT * FROM main_submissions", con)
print "Total number of comments downloaded:", all_comments.shape[0]
print "Total number of submissions downloaded:", all_submissions.shape[0]


con.commit()
cur.close()
con.close()

print 'Runtime: ' + str(int(time.time() - old_time)) + "s"