"""
a separate file for db connection
need to mask the password and user with os.environ["SQ;_USERNAME"]
with a try-except operation
"""

import psycopg2
import os

dbname = 'slack_db'
host = '35.196.43.107'
port = '5432'
password = "ji@hany@n2473"
user = "postgres"
# user = os.environ["SQL_USERNAME"]
# password = os.environ["SQL_PASSWORD"]



# define con and cur here to save future definition
try:
	con = psycopg2.connect(
	   database = dbname,
	   user = user,
	   password = password,
	   host = host,
	   port = port
	)
	con = psycopg2.connect(database="testdb", user="postgres", password="pass123", host="127.0.0.1", port="7890")
	cur = con.cursor()
except:
	print ("connection failure")
