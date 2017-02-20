import os
import matplotlib.image as mpimg
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten

classes = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing','No passing for vehicles over 3.5 metric tons','Right-of-way at the next intersection','Priority road','Yield'
,'Stop','No vehicles','Vehicles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left','Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing',
'End of all speed and passing limits','Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory','End of no passing','End of no passing by vehicles over 3.5 metric tons']

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

x = tf.placeholder(tf.float32, (None, 32, 32, 3)) 
logits = LeNet(x)
saver = tf.train.Saver()


def evaluator(X_data):
    sess = tf.get_default_session()
    return sess.run(tf.argmax(logits, 1), feed_dict= {x: X_data})
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    pictures = []
    for image in os.listdir("testing/"):
        if(image != ".DS_Store"):
            img = "testing/"+image
            im = mpimg.imread(img)
            pictures.append(im)
    labels = evaluator(pictures)
    for i in range(0,len(pictures)):
        plt.figure(figsize=(2,2))
        print(classes[labels[i]])
        plt.imshow(pictures[i])
        plt.show()
    