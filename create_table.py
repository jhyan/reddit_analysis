"""
This module creates the tables in our main SQL database that will store the
data from Reddit (subreddits, submissions and comments).

Three tables are created, namely main_subreddits, main_submissions and main_comments. 
The primary key is their ids. Notice that in table main_comments, the subreddit_id_ 
and submission_id will be serving as foreign keys to the other two tables.
"""

import pandas as pd
from db_connect import *# Connect to SQL

try:
      cur.execute("""
                  CREATE TABLE main_subreddits
                  (
                  id VARCHAR(255) NOT NULL,
                  name TEXT NOT NULL,
                  title TEXT NOT NULL, 
                  created INT NOT NULL,
                  total_submissions INT NOT NULL
                  )
                  """)
except:
      print ("table main_subreddits already exists")



try:
      cur.execute("""
                  CREATE TABLE main_submissions
                  (
                  id VARCHAR(255) NOT NULL,
                  subreddit_id VARCHAR(255) NOT NULL,
                  content TEXT,
                  created INT NOT NULL
                  )
                  """)
except:
      print ("table main_submissions already exists")


try:
      cur.execute("""
                  CREATE TABLE main_comments
                  (
                  id VARCHAR(255) NOT NULL,
                  subreddit_id VARCHAR(255) NOT NULL,
                  submission_id VARCHAR(255) NOT NULL,
                  content TEXT
                  )
                  """)
except:
      print ("table main_comments already exist")


con.commit()

# no close here since con and cur will be used in future
# cur.close()
# con.close()