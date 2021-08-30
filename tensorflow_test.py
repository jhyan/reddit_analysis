
'''
Text classification with Tensorflow: Neuron Network approach
Work-flow: 
split data to train/test set --> gen vocab, total_words and word2index --> 
define instance of data manager --> define graph (input place holder; model; loss, optimizer, accuracy) -->
tune summary graph --> run a session (define epoch, batch) --> check results on tensorboard
'''

# three ways to reduce the overfitting: early stop, l regulerization, dropout


import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
import time
import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

# preprocess and experiments parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # cut off the trivial warning
# learning rate. Test sh and another approach
learning_rate = float(sys.argv[1])
training_epochs = 5
batch_size = 500
display_step = 1
data_cnt = -1  # how many data in Tr and Te. -1 means all
lancaster = nltk.LancasterStemmer()  # apply lancaster stemmer
stopwords = stopwords.words('english')  # 153 stop words. a list.
# network parameters
n_hidden_1 = 200      # 1st layer number of features
n_hidden_2 = 200   # 2nd layer number of features
n_hidden_3 = 200	  # 3rd layer number of features
n_classes = 3         # Categories: graphics, sci.space and baseball
keep_prob = 0.98  # probability the node is kept. For dropout layer
activation_f = tf.nn.relu  # , tf.nn.tanh
beta = 0.0001  # regularization


# helper functions
def alpha_filter(w):
    """
    pattern to match a word of alphabetical characters.
    Can be adjusted to be tolerant to several other characters like ! and digits
    """
    pattern = re.compile(
        '^[!_A-Za-z]+$')  # the ! character is sometimes meaningful
    if pattern.match(w):
        return True
    else:
        return False


def gen_tokens(text):
    '''
    input a text string and return the tokenized stemed list
    '''
    tokens = WordPunctTokenizer().tokenize(text)
    return [lancaster.stem(t) for t in tokens
            if alpha_filter(t) and t.lower() not in stopwords]


def gen_vocab(data1, data2):
    """
    build the vocabulary set
    """
    vocab = []
    for data in [data1, data2]:
        for text in data:
            vocab.extend(gen_tokens(text))
    # freq = Counter(vocab)
    vocab = set(vocab)
    total_words = len(vocab)
    print("vocabulary size: ", len(vocab))  # nearly 20k vocab
    return vocab, total_words


def get_word_2_index(vocab):
    '''
    generate a dictionary of word:idx
    '''
    word2index = {}
    for i, word in enumerate(vocab):  # note the enumerate is not freq, it's 0-N
        word2index[word.lower()] = i
    return word2index


# theoretical instruction:
# http://neuralnetworksanddeeplearning.com/chap3.html
def add_layer(input_tensor, in_size, out_size, activation_f=None):
    # add a layer with defined activation function
    with tf.name_scope('layer'):  # for sketch usage
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(input_tensor, weights) + biases
            wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
        if activation_f:
            output_tensor = activation_f(wx_plus_b, )
        else:
            output_tensor = wx_plus_b
        # import ipdb; ipdb.set_trace(); # one way to debug
        regularizer = tf.nn.l2_loss(weights)
        return output_tensor, regularizer  # use regularizer to break things down in parts


def gen_logfile_name(learning_rate, n_hidden_1, n_hidden_2, n_hidden_3, activation_f, keep_prob):
    """
    generate proper log file name in grid search
    """
    return "lr={0}_n1={1}_n2={2}_n3={3}_af={4}_kp={5}".format(learning_rate, n_hidden_1, n_hidden_2, n_hidden_3, activation_f, keep_prob)


class DataManager(object):
    '''
    manage data and batches
    '''

    def __init__(self, xTr, yTr, xTe, yTe, batch_size):
        self.xTr, self.yTr = xTr, yTr
        self.xTe, self.yTe = xTe, yTe
        self.batch_size = batch_size
        self.batches_in_epoch = int(len(self.xTe)/self.batch_size)

    def get_batch(self, data, target, i, batch_size, total_words, word2index):
        """
        data: content without lable
        target: label
        i is the index of the current batch
        return well-defined numpy 2d 0-1 matrixs
        """
        batches = []
        labels = []
        texts = data[i*batch_size:i*batch_size+batch_size]
        categories = target[i*batch_size:i*batch_size+batch_size]
        for text in texts:
            layer = np.zeros(total_words, dtype=float)
            for word in gen_tokens(text):  # gen_tokens return a list of tokens
                layer[word2index[word.lower()]] += 1
            batches.append(layer)

        for category in categories:  # category is a number here
            y = np.zeros((3), dtype=float)
            y[category] = 1.
            labels.append(y)
        return np.array(batches), np.array(labels)


def main():
    ################## non tf part #################
    print("\nlet's roll")
    old_time = time.time()
    # load sklearn text data
    categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
    newsgroups_train = fetch_20newsgroups(
        subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(
        subset='test', categories=categories)  # <class 'sklearn.utils.Bunch'>
    train_data, train_target = newsgroups_train.data[:
                                                     data_cnt], newsgroups_train.target[:data_cnt]
    test_data, test_target = newsgroups_test.data[:
                                                  data_cnt], newsgroups_test.target[:data_cnt]
    print('total texts in train:', len(train_data))
    print('total texts in test:', len(test_data))
    # gen vocab, total_words and worddict
    vocab, total_words = gen_vocab(train_data, test_data)
    word2index = get_word_2_index(vocab)
    n_input = total_words  # Words in vocab
    # define instance of data manager
    dm = DataManager(train_data, train_target,
                     test_data, test_target, batch_size)

    ###################  tf part ###################
    with tf.Graph().as_default():
        # input place holder
        with tf.name_scope('inputs'):
            # place holder defines the dimension of tensor
            input_tensor = tf.placeholder(
                tf.float32, [None, n_input], name="input")
            output_tensor = tf.placeholder(
                tf.float32, [None, n_classes], name="output")

        # construct model
        out1, r1 = add_layer(input_tensor, n_input,
                             n_hidden_1, activation_f=activation_f)
        out2, r2 = add_layer(out1, n_hidden_1, n_hidden_2,
                             activation_f=activation_f)
        out3, r3 = add_layer(out2, n_hidden_2, n_hidden_3,
                             activation_f=activation_f)
        # no activation for the output layer
        prediction, r4 = add_layer(out3, n_hidden_3, n_classes)
        # define loss, optimizer, accuracy
        with tf.name_scope('loss'):
            # Loss function with L2 Regularization with beta=0.01
            regularizers = r1 + r2 + r3  # r4 cannot be added
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction, labels=output_tensor))  # <class 'tensorflow.python.framework.ops.Tensor'>
            loss = tf.reduce_mean(loss + beta * regularizers)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss)  # <class 'tensorflow.python.framework.ops.Operation'>
        with tf.name_scope('Accuracy'):
            # argmax is used since the logits
            correct_prediction = tf.equal(
                tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
            # 1D tensor --> 1 float vector --> calculate the rate e.g. 7/10
            # Casts a tensor to a new type. Tensor("Mean_1:0", shape=(), dtype=float32)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # merge all summaries into a single "operation" which we can execute in a session
        tf.summary.scalar("cost", loss)
        tf.summary.scalar("accuracy", accuracy)
        summary_op = tf.summary.merge_all()
        ###### end of merge summary #######################

        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        with tf.Session() as sess:
            # create log writer object and init session
            tail = gen_logfile_name(learning_rate, n_hidden_1, n_hidden_2, n_hidden_3, str(
                activation_f).split()[1], keep_prob)
            train_writer = tf.summary.FileWriter(
                "logs/train/" + tail, sess.graph)
            test_writer = tf.summary.FileWriter(
                "logs/test/" + tail, sess.graph)
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(train_data)/batch_size)
                # Loop over all batches
                for i in range(dm.batches_in_epoch):
                    batch_xTr, batch_yTr = dm.get_batch(
                        dm.xTr, dm.yTr, i, dm.batch_size, total_words, word2index)
                    # print (batch_xTr.shape, batch_yTr.shape)

                    # Training data. Run cost, summary, optimizer
                    cTr, _, train_summary = sess.run([loss, optimizer, summary_op], feed_dict={
                                                     input_tensor: batch_xTr, output_tensor: batch_yTr})
                    avg_cost += cTr / total_batch
                    train_writer.add_summary(
                        train_summary, epoch * dm.batches_in_epoch + i)  # subtle equation

                    # Testing data. Cut the optimizer to avoid training while evaluating!!
                    batch_xTe, batch_yTe = dm.get_batch(
                        dm.xTe, dm.yTe, 0, len(dm.xTe), total_words, word2index)
                    cTe, test_summary = sess.run([loss, summary_op], feed_dict={
                                                 input_tensor: batch_xTe, output_tensor: batch_yTe})
                    test_writer.add_summary(
                        test_summary, epoch * dm.batches_in_epoch + i)

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "loss=",
                          "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
        print('Runtime: ' + str(int(time.time() - old_time)) + "s")


if __name__ == "__main__":
    main()


############# explanations of some methods ###########################
# reduce_mean: calculate mean
# eval: load xTe and yTe return accuracy
# argmax: use with the logits (a bag of possibilities)
# cast: transfer a tensor to another type
# tensor.eval equals running the default session
# place_holder: virtual container for a tensor
# A Tensor object is a symbolic handle to the result of an operation,
# but does not actually hold the values of the operation's output.


############# backup codes to understand graph and session #############
# graph = tf.Graph()
# with graph.as_default():
# 	variable = tf.Variable(42, name='foo')
# 	initialize = tf.global_variables_initializer()
# 	assign = variable.assign(13)

# with tf.Session(graph=graph) as sess:
# 	sess.run(initialize)
# 	sess.run(assign)
# 	print"initilize and assign: ", sess.run(variable)
# # Output: 13

# with tf.Session(graph=graph) as sess:
# 	sess.run(initialize)
# 	print "initilize only: ", sess.run(variable)
# # Output: 42


################ remaining IQ challenge ###############
# how to implement the product mudule with the least memory
