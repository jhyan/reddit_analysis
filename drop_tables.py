"""
a separate file that drops all tables in the db (cleans the db).
"""

import os
import pandas as pd
import praw
import sys
import time
from datetime import datetime
from db_connect import *
# drop all the tables to prevent duplicate data
# if it takes too long for your mac do:
# 1. brew services start postgresql
# 2. brew services stop postgresql
# 3. dropdb testdb
cur.execute("SELECT table_schema,table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_schema,table_name")
rows = cur.fetchall()
for row in rows:
    print("dropping table:", row[1])   
    cur.execute("drop table " + row[1] + " cascade") 
    print("finish one")   
# cur.close()
con.commit()
