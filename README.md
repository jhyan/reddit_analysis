
# Ptah

#### Table of Contents
* [Presentation](#presentation)
* [Installation](#installation)
* [Running](#running)
    * [Reddit scraping tool](#reddit-st)
    * [Corpus correlation analysis](#Corpus-correlation)
    * [Reddit recommendation system](#Recommendation-system)
    * [Tensorflow-analysis](#tensorflow-analysis)
* [Results](#results)
* [Pretrained model](#pretrained-model)
* [FUTURE](#future)
* [Side notes](#side-notes)
* [Licenses](#license)

## presentation
This work tries to do English text classification with TF-IDF and Neuron Network (NN). The goal is to train models from reddit corpus and test on slack conversations. The logic behind it is to classify and analyze conversations on chatting apps for marketing. It's done in python and tensorflow. There are three modules in the reposoitory:
1. __Reddit scraping tool__. Scrap the reddit data in the hirerarchy of _subreddit--post--comments_ and saves to the database.
1. __Corpus correlation analysis__. Pulls data from database and runs a tfidf analysis of different subreddits. Generates correlation graphs.
1. __Reddit recommendation system__. Taken into consideration the tfidf score, post time, post length, upvotes, word tag amplification, to recommend a most related reddit submission.
1. __Tensorflow test__. Build a neuron netword model for text classification. Generates logs for tensorboard visualization.

## installation
FIrst, set the virtual environment (e.g. https://programwithus.com/learn/python/pip-virtualenv-mac)

Second, execute the below bash code.
```bash
pip install -r requirement.txt
```

If there're errors, follow the below sections:

Install PostgreSQL:
1.  Install postgres app https://postgresapp.com/downloads.html
1.  export PATH=$PATH:/Applications/Postgres.app/Contents/Versions/latest/bin
1.  pip install psycopg2
1.  brew install postgresql
1.  brew services start postgresql

After PostgreSQL installation, for the first time to config the postgreSQL db server, in terminal do:
```bash
psql postgres
```

Then do 
```bash
CREATE ROLE postgres WITH LOGIN PASSWORD 'pass123';
ALTER ROLE postgres CREATEDB;
CREATE DATABASE testdb;
```

Notice that during development process, if you want to drop the tables to prevent duplicate data and it takes too long for running the drop_tables.py, do:

```bash
brew services stop postgresql
brew services start postgresql
dropdb testdb
```

After dropping the table, to recreate again:
```bash
psql postgres
```
Then do 
```base
CREATE DATABASE testdb;
```

## running
### reddit-st
Run the below:
```bash
python get_reddit_data.py
```

### slack-st
First, print bot id to visualize.
```bash
export SLACK_BOT_TOKEN='xoxb-238963196967-cUBXeOoXlfZypALVDLA7LsMJ'
python print_bot_id.py
```
Second, use the printed bot-ID to run the slack-bot.
```bash
export BOT_ID='U70UB5SUF' 
python slack_bot.py
```
Tht bot should be running actively now on slack and listening to Sympler's dewvelopment team's conversation.

### Corpus-correlation
Run the below code in bash:
```bash
python subreddits_analysis.py
```

### Recommendation-system
Run the below code in bash:
```bash
python submissions_analysis.py
```

### tensorflow-analysis
First, run the below code in bash:
```bash
sh ./test.sh -l 0.01
```
Optional parameters table:

|argument name|meaning|imperical values
|:--------:|:-------:|:------:
|-l|learning rate|0.01~0.03
|-b|regularization parameter|0.01-0.1
|-p|keep probability (for dropout)|0.95~0.99
|more to be added||

Second, run the bash code:
```bash
tensorboard --logdir='logs/'
```

### results
__reddit scraping tool__:
The tool scrapes reddit following a hierarchy of subreddit--submissions--comment. The logic behind it is to store info of subreddit first into the db. Then use the subreddit info in db to obtain the lower layers: posts and comments.

File details:
_dbconnect_: try to connect with the cloud psql database. Can also connect to local db. A try-except is applied to inspect the failure of connection. The file would yield a db cursor (variable _cur_) and imported into the _create_table_ file.
_create_table_: three tables are created, namely main_subreddits, main_submissions and main_comments. The primary key is their ids. Notice that in table _main_comments_, the _subreddit_id_ and _submission_id_ will be serving as foreign keys to the other two tables.
_drop_tables_: a separate file that drops all tables in the db (cleans the db).
_get_reddit_data_: Scrape the reddit data and save to db. Implemented by three layered for loops and praw api.


__slack scraping tool__:
After running, there will be a bot running in the backend listening to the channel assigned. It has the ability to deal with both the real time interaction info and the history info.

File details:
_print_bot_id.py_: a snippet to print the slack bot id given the name of the bot.
_slack_bot.py_: a while-true loop utilizing the rtm(real-time-messaging) api is able to read the real time conversation. The text history is fetched into chunks and parsed using regular expression. The message history is stored in _slack_history.txt_.


__corpus correlation analysis__:
<a href="url"><img src="normalized_correlation.png" align="middle" width="700" ></a>

The darker color, the more correlation. Three observations are made:
1. Python is more related to data science than Java
2. Fitness is not relative to any other topics (<0.5 correlation)
3. Programming languages are clustered and cs fields are clustered.
4. After adding the synonyms of words into tokens, the result is getting more accurate.

The _Stranger things_ related subreddits also show the similar result. 

File details:
_subreddits_analysis.py_: it compares relations between different subreddits. 
Simple workflow: db data --> panda frame --> token, stem add synonym --> 
tfidf vector --> fit transform docs --> seaborn module draw figure --> snakeviz see profile

__reddit recommendation system__:
The most related posts will be shown as the below format:
Question 1: Who dances best Bob or Hopper?  # the raw question
|---id-----|-length-|--score-|---time-------|---url---------------# title of each entity
['7dkcii', '0325', 0.2842, 1510947712, 'url: redd.it/7dkcii'] # first filtering with nlp score
['5ykoji', '3138', 0.0239, 1489157496, 'url: redd.it/5ykoji']
['7azrpq', '1926', 0.0228, 1509939679, 'url: redd.it/7azrpq']
+++++finished select nlp score\++++++
['5ykoji', ' 3138', 0.0239, 1489157496, 'url: redd.it/5ykoji']
['7azrpq', ' 1926', 0.0228, 1509939679, 'url: redd.it/7azrpq']
+++++finished select length+++++ # second filtering with length
['7azrpq', ' 1926', 0.0228, 1509939679, 'url: redd.it/7azrpq']
+++++finished select time+++++ # third filtering with the created time
Final match id for question: Who dances best Bob or Hopper?
redd.it/7azrpq # final recommendation result for the question
Runtime till now: 14s # run time stats

The simple flow: db data --> panda frame --> gen docs (token, stem add synonym) --> add question to the docs--> get cosine of rows in tfidf matrix --> filter by time/length/score --> save the result to a local file.

__tensorflow analysis__:
The tensorflow analysis now takes the sklearn news corpus as the input. The categories are _comp.graphics, sci.space_ and _rec.sport.baseball_. Since the code has the ability to do logs for different training/testing steps, it's easier to do bias-variance trade off analysis.

<a href="url"><img src="tf_learning-rate 0.01 and 0.03.png" align="middle" width="700" ></a>

Observations:
1. In this scope of training/testing dataset, 0.03 learning rate has less bias and also less variance.
2. Regularizations/early stop/dropout helps to reduce bias.
3. Adding more data helps or not is a criteria.

Several tips:
1. low bias, high variance is the common result of NN
2. Less layers/nodes are, less initial loss. On contrast, larger, steep loss cliff
3. NN's are usually used with large datasets where CV is not used - and very expensive
4. Cut off optimizer during epoch evaluation. Carefully run with xTr and xTe
5. May consider bagging (random forest) and boosting (adaboost).

File details:
split data to train/test set --> gen vocab, total_words and word2index --> 
define instance of data manager --> define graph (input place holder; model; loss, optimizer, accuracy) --> tune summary graph --> run a session (define epoch, batch) --> check results on tensorboard

## Future
1. Multi-thread distributed may be considered for scaling. Multithreads may help if certain tasks are blocked so that others are waiting for it.
2. Tennsorflow runs in GPU and distributed calculation. If the dataset goes large it may be neeeded.
3. How to see it as a product and where to position in the Sympler product design line.

## side-notes
reddit bot:
* client_id='qrbjP0JQu3Uo1Q',
* client_secret='lyFcQOLLN1OeH1a4-BE5WIJeWgM',
* user_agent = 'doe nlp usage',
* username='sisyphus_bot',
* password='Sympler1~'