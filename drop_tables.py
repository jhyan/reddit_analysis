"""
a separate file that drops all tables in the db (cleans the db).
"""

import os
import pandas as pd
import praw
import sys
import time
from datetime import datetime
# from create_table import *
from db_connect import *
# drop all the tables to prevent duplicate data
cur.execute("SELECT table_schema,table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_schema,table_name")
rows = cur.fetchall()
print rows
for row in rows:
    print "dropping table:", row[1]
    cur.execute("drop table " + row[1] + " cascade")
    print 'finish one'
con.commit()