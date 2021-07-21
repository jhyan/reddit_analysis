'''
A slack bot to inspect the conversations ans archieve history messages.

A while-true loop utilizing the rtm(real-time-messaging) api is able to 
read the real time conversation. The text history is fetched into chunks 
and parsed using regular expression. The message history is stored in _slack_history.txt_.
'''

from __future__ import division
import os
import time
from slackclient import SlackClient
import math
import re


# export BOT_ID='U70UB5SUF' # starterbot's ID as an environment variable
BOT_ID = os.environ.get("BOT_ID")
# BOT_ID='U70UB5SUF'

# constants
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "do"
HISTORY_CNT = 1000

# instantiate Slack 
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))


TARGET_CHANNEL = "development" # which channel to fecth info from
# TARGET_CHANNEL = "product-design"

def handle_command(command, channel):
    """
    Receives commands directed at the bot and determines if they
    are valid commands. If so, then acts on the commands. If not,
    returns back what it needs for clarification.
    """
    response = "Not sure what you mean. Use the *" + EXAMPLE_COMMAND + \
               "* command with numbers, delimited by spaces."
    if command.startswith(EXAMPLE_COMMAND):
        response = "Sure...write some more code then I can do that!"
    # chat.postMessage is the key function
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)

def parse_slack_output(slack_rtm_output):
    """
    The Slack Real Time Messaging API is an events firehose.
    this parsing function returns None unless a message is
    directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        # print ("the whole output list is: ", output_list)
        for output in output_list: # output list is a list and output is a dictionary
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, so split with the AT_BOT
                print ("output text: ", output['text']) # e.g. '<@U70UB5SUF> yo man hey'
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']
    return None, None


def list_channels():
    '''
    returns a dictionary of channel name: channel id
    '''
    channels_call = slack_client.api_call("channels.list") # lazy method!
    lookup = {}
    if channels_call.get('ok'): # return call status
        channels =  channels_call['channels']
        if channels:
            for c in channels:
                lookup[c['name']] = c['id']
        else:
            print("Unable to authenticate.")

    return lookup


def parse_history_dict(history):
    """
    channels.history returns a dictionary
    parse the channels.history and return a list of strings
    """
    ret, oldest_ts = [], None
    for i, h in enumerate(history["messages"]):
        # h["messages"] example: 
        # { u'text': u'Sure...write some more code then I can do that!', \
        #   u'type': u'message', u'user': u'U70UB5SUF', \
        #   u'ts': u'1504884743.000327', u'bot_id': u'B700A8R53'}

        # replace bot name, emoji, wierd coding, and urls
        msg = re.sub("<[/@0-9a-zA-Z].*> ?|:[a-z_].+:",
                    "", h["text"]) # genius regular expression
        msg = re.sub("\.+ *|,+ *|!+ *|\?+ *", "\n", msg)
        ret.append(msg)
        if i == len(history)-1:
            oldest_ts = h['ts'] 
    print "ret: ", ret
    return ret, oldest_ts


if __name__ == "__main__":

    channels = list_channels() # returns a dictionary
    print "channels: ", channels[TARGET_CHANNEL]

    # takea advantage of the latest and oldest timestamp to achieve more chunks
    chunks, rest = int(math.ceil(HISTORY_CNT/1000)), HISTORY_CNT%1000 # max capacity is 1000
    print chunks, rest
    history_lst, oldest_ts = [], time.time() # default to be now not None
    for i in range(chunks):
        if i == chunks-1 and rest != 0:
            chunk_sz = rest
        else:
            chunk_sz = 1000
        history_dict = slack_client.api_call(
            "channels.history", 
            channel = channels[TARGET_CHANNEL], 
            count = chunk_sz,
            latest = oldest_ts
            ) 
        history_lst_chunk, oldest_ts = parse_history_dict(history_dict)
        print "chunk {0} size {1}".format(i, len(history_lst_chunk))
        history_lst.extend(history_lst_chunk)
        # print history_lst, oldest_ts
    print "length of fetched history: ", history_lst  
    print "save files"
    with open('slack_history.txt', 'w') as f:
        for line in history_lst:
            try:
        # content = '\n'.join(history_lst)
                f.write(line)
            except:
                "one line not ascii"
                continue


    # deal with all user interaction
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    if slack_client.rtm_connect(): # real time messaging
        print("StarterBot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read()) # slack_client.rtm_read() is the key function
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
