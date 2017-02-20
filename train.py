import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten


training_file = 'train.p' 
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

print('successfully imported')
# This is the preprocessing step. Training data is being shuffled so that the data is not skewed.
X_train, y_train = shuffle(X_train, y_train)
# I'm allocating 20% of the Training data to cross validation set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

EPOCHS = 50 # Epoch is the number of forward and backward pass
BATCH_SIZE = 64 # Model will consider 64 images at a time

# LeNet-5 model takes in an image of size 32x32x3 (3 represents the depth (R, G, B)) and outputs logits. This model is based on the original architecture proposed by Yann LeCun.
# LeNet architecture is inspired by the biological working of visual cortex. More information on how LeNet works is provided in the document.
def LeNet(x):    
    mu = 0
    sigma = 0.1
    
    conv_layer_1_weights = tf.Variable(tf.truncated_normal(shape = (5,5,3,6), mean = mu, stddev = sigma))
    conv_layer_1_bias = tf.Variable(tf.zeros(6))
    conv_layer_1 = tf.nn.conv2d(x, conv_layer_1_weights, strides=[1, 1, 1, 1], padding='VALID') + conv_layer_1_bias
    conv_layer_1 = tf.nn.relu(conv_layer_1)
    conv_layer_1 = tf.nn.max_pool(conv_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv_layer_2_weights = tf.Variable(tf.truncated_normal(shape = (5,5,6,16), mean = mu, stddev = sigma))
    conv_layer_2_bias = tf.Variable(tf.zeros(16))
    conv_layer_2 = tf.nn.conv2d(conv_layer_1, conv_layer_2_weights, strides = [1,1,1,1], padding = 'VALID') + conv_layer_2_bias
    conv_layer_2 = tf.nn.relu(conv_layer_2)
    conv_layer_2 = tf.nn.max_pool(conv_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    flat = flatten(conv_layer_2)

    fully_connected_1_weights = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fully_connected_1_bias = tf.Variable(tf.zeros(120))
    fully_connected_1 = tf.matmul(flat, fully_connected_1_weights) + fully_connected_1_bias
    fully_connected_1 = tf.nn.relu(fully_connected_1)

    fully_connected_2_weights = tf.Variable(tf.truncated_normal(shape = (120, 84), mean = mu, stddev = sigma))
    fully_connected_2_bias = tf.Variable(tf.zeros(84))
    fully_connected_2 = tf.matmul(fully_connected_1, fully_connected_2_weights) + fully_connected_2_bias
    fully_connected_2 = tf.nn.relu(fully_connected_2)       

    fully_connected_3_weights = tf.Variable(tf.truncated_normal(shape = (84,43), mean = mu, stddev = sigma))
    fully_connected_3_bias = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fully_connected_2, fully_connected_3_weights)
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3)) # I'm using None because we don't the number of examples present
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43) # 43 corresponds to the number of classes

rate = 0.001 #learning rate

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y) # softmax returns the probabilites of each class
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate) # Gradient Descent is another alternative for Adam Optimizer
training_operation = optimizer.minimize(loss_operation) # backward propagation for adjusting weights of the network

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1)) #tf.argmax returns the index with highest value
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initializing all the weights randomly. These random weights are picked from a normal distribution 
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet') # Saving the model so that weights don't have to be initialized again
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))            




