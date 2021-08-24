"""
Package for the PosrtgreSql db connection.
"""

import psycopg2
import os

try:
	print ("starting to connect......")
	con = psycopg2.connect(database="testdb", user="postgres", password="pass123", host="127.0.0.1", port="5432")
	cur = con.cursor()
	print ("Databse connected!")
except Exception, err:
	print err
