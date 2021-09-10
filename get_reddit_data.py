'''
    Scrape the reddit data and save to the database.
    '''

import os
import pandas as pd
import praw
import sys
import time
from datetime import datetime
import concurrent.futures
from concurrent.futures import ALL_COMPLETED
from drop_tables import *
from create_table import *
import threading

HOT_CNT = 10
COMMENTS_THRESHOLD = 50 # the smaller, the faster, but with less prediction accuracy

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
print(("I am reddit bot {0}".format(reddit.user.me())))

# chosen_subreddits = [ 'Fitness', 'java', 'python','datascience', 'MachineLearning']
chosen_subreddits = ['StrangerThings', 'TheUpsideDown', 'Stranger_Things', 'netflix']
""" for i in range(len(chosen_subreddits)):
    subreddit = reddit.subreddit(chosen_subreddits[i])
    db_tuple = (subreddit.id, chosen_subreddits[i], subreddit.title, int(subreddit.created), 0)
    print (db_tuple)
    sql_query = INSERT INTO main_subreddits VALUES (%s, %s, %s, %s, %s)
    cur.execute(sql_query, db_tuple) # use tuple to insert """

#The i stands for the index and we expect this function to insert the subreddit data to the database.
def insert_subreddits(i):
    subreddit = reddit.subreddit(chosen_subreddits[i]) # get subreddit by a string name
    #scrape all the posts and their comments from a list of subreddits
    # store subreddit data into the database
    db_tuple = (subreddit.id, chosen_subreddits[i], subreddit.title, int(subreddit.created), 0)
    print(db_tuple)
    sql_query = """INSERT INTO main_subreddits VALUES (%s, %s, %s, %s, %s)"""
    cur.execute(sql_query, db_tuple) # use tuple to insert



# Note: happens in 1 thread so the comments_cnt limit is still effective
def insert_to_db(subr, subreddit_id, comments_cnt):
    # deal with submissions
    for submission in subr.hot(limit=HOT_CNT):  # this step connects the internet?
        submission.comments.replace_more(limit=0)  # For deep comments
        submission_tuple = (submission.id, subreddit_id, submission.selftext, int(submission.created))
        submission_query = "INSERT INTO main_submissions VALUES (%s, %s, %s, %s)"
        cur.execute(submission_query, submission_tuple)
        # deal with comments
        comment_tuples = []
        comment_list = submission.comments.list()
        for comment in comment_list:
            comment_tuples.append((comment.id, subreddit_id,
                                   submission.id, comment.body))
        if len(comment_tuples) > 0:
            comment_str = b','.join(cur.mogrify("(%s,%s,%s,%s)", x) for x in comment_tuples).decode()
            cur.execute("INSERT INTO main_comments VALUES " + comment_str)
        comments_cnt += len(comment_list)
        if comments_cnt > COMMENTS_THRESHOLD:
            break

def gen_args_list(subreddits_df):
    res = []
    # Parse subreddits-->submissions-->comments and save to SQL
    for idx in range(subreddits_df.shape[0]):
        # deal with subreddit
        args = (reddit.subreddit(subreddits_df['name'][idx]), subreddits_df['id'][idx], 0)
        res.append(args)
    return res

# multi-threading, more see https://rednafi.github.io/digressions/python/2020/04/21/python-concurrent-futures.html
# threading difficulties:
# 1. Identify IO/CPU bottleneck
# 2. How to avoid data overflow when developing (avoid database explosion)
# 3. How to apply restrictions within threading
# 4. Thread on which layer

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    t_time = time.time()
    result_subreddits = {executor.submit(insert_subreddits, i): i for i in range(len(chosen_subreddits))}
    concurrent.futures.wait(result_subreddits, return_when=ALL_COMPLETED)
    print(('Runtime for insert_subreddits: ' + str(int(time.time() - old_time)) + "s"))
    subreddits_df = pd.read_sql("SELECT * FROM main_subreddits", con) # <clddlass 'pandas.core.frame.DataFrame'>
    print(subreddits_df)
    args_list = gen_args_list(subreddits_df)
    future_results = {executor.submit(insert_to_db, *arg): arg for arg in args_list}
    for future in concurrent.futures.as_completed(future_results):
        try:
            _ = future.result()
        except Exception as exc:
            print('generated an exception: %s' % exc)


#Check the total amount of submissions and comments downloaded. It's rolling.
all_comments = pd.read_sql("SELECT * FROM main_comments", con)
all_submissions = pd.read_sql("SELECT * FROM main_submissions", con)
print(("Total number of comments downloaded:", all_comments.shape[0]))
print(("Total number of submissions downloaded:", all_submissions.shape[0]))


con.commit()
cur.close()
con.close()

print(('Runtime: ' + str(int(time.time() - old_time)) + "s"))
