'''
Scrape the reddit data and save to db.
Implemented by three layered for loops and praw api.
'''

import os
import pandas as pd
import praw
import sys
import time
from datetime import datetime

from praw.reddit import Submission
from create_table import *
from db_connect import *
import concurrent.futures


old_time = time.time()
HOT_CNT = 100

# some helper functions
def convert_secs(secs_float):
	"""
	Takes a number of seconds and returns datetime
	"""
	return datetime.fromtimestamp(secs_float).strftime("%B %d, %Y %I:%M:%S")


comments_threshold = 5000 # global
reddit = praw.Reddit(client_id='qrbjP0JQu3Uo1Q',
                     client_secret='lyFcQOLLN1OeH1a4-BE5WIJeWgM',
                     user_agent = 'doe nlp usage',
                     username='sisyphus_bot',
                     password='Sympler1~') # user_agent is just a description
print("I am slock bot {0}".format(reddit.user.me()))


# only after con.commit can all the sql causes be valid
# chosen_subreddits = [ 'Fitness', 'java', 'python','datascience', 'MachineLearning']
chosen_subreddits = ['StrangerThings', 'TheUpsideDown', 'Stranger_Things', 'netflix']

def insert_subreddits(i):
    subreddit = reddit.subreddit(chosen_subreddits[i]) # get subreddit by a string name 
    #scrape all the posts and their comments from a list of subreddits
    # store subreddit data into the database
    db_tuple = (subreddit.id, chosen_subreddits[i], subreddit.title, int(subreddit.created), 0)
    print(db_tuple)
    sql_query = """INSERT INTO main_subreddits VALUES (%s, %s, %s, %s, %s)"""
    cur.execute(sql_query, db_tuple) # use tuple to insert


# # the below would have different outputs
# def f(x):
#     print x**2
# pool = Pool(3)
# result = pool.map(f, [1,2,3,4,5])
# pool.close()
# pool.join()



############# database is down ##############
# subreddit -> pandas dataframe
pull_df = pd.read_sql("SELECT * FROM main_subreddits", con) # <clddlass 'pandas.core.frame.DataFrame'>

#global
submissions_cnt = 0
comment_count = 0


# Parse subreddits-->submissions-->comments and save to SQL
def get_subreddit(index):
    print ("Subreddit " + str(index + 1) + " / " + 
    	   str(pull_df.shape[0]) + ": " + pull_df['title'][index])
    # deal with subreddit
    subreddit = reddit.subreddit(pull_df['name'][i])
    subreddit_id = pull_df['id'][index]
    return (subreddit,subreddit_id)

def get_submission(combine):  #index 0 is submission index 1 is subreddit_id
    
    global submissions_cnt
    submissions_cnt += 1

    submission = combine[0]
    subreddit_id = combine[1]
    submission.comments.replace_more(limit = 0); # For deep comments
    submission_tuple = (submission.id, subreddit_id, submission.selftext, int(submission.created))
    submission_query =  "INSERT INTO main_submissions VALUES (%s, %s, %s, %s)"
    cur.execute(submission_query, submission_tuple)
    # deal with comments
    comment_list = submission.comments.list()


    return (comment_list,submission)

def get_comment(combine):
    comment_list = combine[0]
    subreddit_id = combine[2]
    submission = combine[1]
    comment_tuples = []
    for comment in comment_list:
        global comment_count
        comment_count+=1

        # Exit if you've downloaded more than comments_threshold comments
        if comment_count > comments_threshold: break 
        comment_tuples.append((comment.id , subreddit_id, 
                                submission.id, comment.body))

    if len(comment_tuples) > 0: # smart, insert in one sentense
            comment_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x) for x in comment_tuples)
            cur.execute("INSERT INTO main_comments VALUES " + comment_str)

    
    # The rest is for pretty tracking of progress
    sys.stdout.write("\rComments processed: {0}, Submissions processed: {1}".format(comment_count, submissions_cnt)) # print i, comma means wait for all to finish. \r and \n cannot be used together
    sys.stdout.flush()
    
   
if __name__ == "__main__":
    with concurrent.features.ThreadPoolExecutor() as executor:
        executor.map(insert_subreddits,[i for i in range(len(chosen_subreddits))])

    t_start = time.time()
    with concurrent.features.ThreadPoolExecutor() as executor:
        result1 = executor.map(get_subreddit,[i for i in range(pull_df.shape[0])])

    for result in result1:
        submissions = result[0].hot(limit=HOT_CNT)  # this step connects the internet? #result[0] is subreddit
        with concurrent.features.ThreadPoolExecutor() as executor:
            result2 = executor.map(get_submission,[(submission,result[1]) for submission in submissions]) #result[1] is subreddit_id

        for tup in result2:
            
            comment_list = tup[0] #tup[0] is comment_list tup[1] is submission
            with concurrent.features.ThreadPoolExecutor() as executor:
                result3 = executor.map(get_comment,[(comment_list,tup[1],result[1])])
    print('  Runtime: ' + str(int(time.time() - t_start)) + "s")
    print('-' * 50)

    #Check the total amount of submissions and comments downloaded. It's rolling.
    all_comments = pd.read_sql("SELECT * FROM main_comments", con)
    all_submissions = pd.read_sql("SELECT * FROM main_submissions", con)
    print("Total number of comments downloaded:", all_comments.shape[0])
    print("Total number of submissions downloaded:", all_submissions.shape[0])


    con.commit()
    cur.close()
    con.close()

    print('Runtime: ' + str(int(time.time() - old_time)) + "s")