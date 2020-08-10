#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IMPORTS
import tensorflow as tf
import numpy as np
from skimage import transform
import os
from skimage.color import rgb2gray
import skimage.io as imd
import matplotlib.pyplot as plt
import random
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess=tf.compat.v1.Session()


def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, d))]

    labels = []
    images = []

    # ahora vamos a las imagenes
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith('.ppm')]

        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


images_train, labels_train = load_ml_data('BelgiumTSC_Training/Training')
images_test, labels_test = load_ml_data('BelgiumTSC_Testing/Testing')
print('Number of images:', images_train.size)
print('Number of labels:', labels_train.size)
print('Number of distinc traffic signals:', len(set(labels_train)))


plt.title('Cuantity of traficcs signals per label')
plt.hist(labels_train, len(set(labels_train)))
pass


rand_signals = random.sample(range(0, len(labels_train)), 10)
rand_signals

#Print some images
def rand_signals_plot(array, number_sample, color='jet'):
    '''
    To represent random traffic signals
    :param array: an array with the traffic signals images
    :param number_sample: number of samples to plot
    :param color: the color to show the images, default is RGB, but you can choose gray or another
    :return: a plot of sample traffic signals from your dataset
    '''
    rand_signals = random.sample(range(0, len(array)), number_sample)
    for n in range(number_sample):
        plt.figure(figsize=(14,10))
        temp_img = array[rand_signals[n]]
        plt.subplot(1, 10, n+1)
        plt.axis('off')
        plt.imshow(temp_img, cmap=color)
        plt.show()
        print('image structure:{}'.format(temp_img.shape))
        
        
rand_signals_plot(images_train, 10)


#Show all distinct images
unique_labels = set(labels_train)
plt.figure(figsize=(16,16))
i = 1
for label in unique_labels:
    temp_im = images_train[list(labels_train).index(label)]
    plt.subplot(8,8, i)
    plt.axis("off")
    plt.title("Class {0} ({1})".format(label, list(labels_train).count(label)))
    i +=1
    plt.imshow(temp_im)
pass


w = 999
h = 999

for image in images_train:
    if image.shape[0] < h:
        h = image.shape[0]
    if image.shape[1] < w:
        w = image.shape[1]
print('Min size:{}x{}'.format(h,w))


#Resize images
images_train_28 = [transform.resize(image, (28,28)) for image in images_train]
images_test_28 = [transform.resize(image, (28,28)) for image in images_test]


# Converto into gray scale
images_train_28 = np.array(images_train_28)
images_train_28_gray = rgb2gray(images_train_28)

images_test_28 = np.array(images_test_28)
images_test_28_gray = rgb2gray(images_test_28)

rand_signals_plot(images_train_28_gray, 10, color='gray')


## red neuronal 

ops.reset_default_graph()
sess = tf.compat.v1.Session()

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = images_train_28_gray[0].shape[0]
image_height = images_train_28_gray[0].shape[1]
flat_image_shape = images_train_28_gray[0].shape[0]*images_train_28_gray[0].shape[1]
target_size = max(labels_train)+1
num_chanels = 1
generations = 800
eval_every = 25
full_connected_size1 = 50
full_connected_size2 = 100

tf.compat.v1.disable_eager_execution()
x_input_shape = (batch_size, image_width, image_height, num_chanels)
x_input = tf.compat.v1.placeholder(tf.float32, shape = x_input_shape)
y_target = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))

eval_input_shape = (None, image_width, image_height, num_chanels)
eval_input = tf.compat.v1.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.compat.v1.placeholder(tf.float32, shape = (evaluation_size))


nn_weight1 = tf.Variable(tf.compat.v1.truncated_normal([flat_image_shape, full_connected_size1], stddev=0.1, dtype=tf.float32))
nn_bias1 = tf.Variable(tf.compat.v1.truncated_normal([full_connected_size1], stddev=0.1, dtype = tf.float32))

nn_weight2 = tf.Variable(tf.compat.v1.truncated_normal([full_connected_size1, full_connected_size2], stddev=0.1, dtype=tf.float32))
nn_bias2 = tf.Variable(tf.compat.v1.truncated_normal([full_connected_size2], stddev=0.1, dtype=tf.float32))

nn_weight3 = tf.Variable(tf.compat.v1.truncated_normal([full_connected_size2, target_size], stddev=0.1, dtype=tf.float32))
nn_bias3 = tf.Variable(tf.compat.v1.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def my_fully_connected(x_input):
    #flatten
    images_flat = tf.compat.v1.layers.flatten(x_input)
    
    #first hidden layer
    layer1 = tf.add(tf.matmul(images_flat, nn_weight1), nn_bias1)
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, 0.8)
    
    #Second hidden layer
    layer2 = tf.add(tf.matmul(layer1, nn_weight2), nn_bias2)
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, 0.8)
    
    #Third layer
    layer3 = tf.add(tf.matmul(layer2, nn_weight3), nn_bias3)
    
    return layer3

model_ouput = my_fully_connected(x_input)
test_model_output = my_fully_connected(eval_input)

## 

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_ouput, labels = y_target))

prediction = tf.nn.softmax(model_ouput)
test_prediction = tf.nn.softmax(test_model_output)

my_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step = my_optim.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis = 1)
    num_corrects = np.sum(np.equal(batch_predictions, targets))
    return 100.0*num_corrects/batch_predictions.shape[0]

train_loss = []
train_acc = []
test_acc = []
i_vals = []

for i in range(generations):
    rand_idx = np.random.choice(len(images_train_28_gray), size = batch_size)
    rand_x = images_train_28_gray[rand_idx]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = labels_train[rand_idx]
    train_dict = {x_input:rand_x, y_target:rand_y}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)
    
    if(i+1) % eval_every == 0:
        rand_idx_eval = np.random.choice(len(images_test_28_gray), size = evaluation_size)
        rand_x_eval = images_test_28_gray[rand_idx_eval]
        rand_x_eval = np.expand_dims(rand_x_eval, 3)
        rand_y_eval = labels_test[rand_idx_eval]
        test_dict = {eval_input:rand_x_eval, eval_target:rand_y_eval}
        temp_test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(temp_test_preds, rand_y_eval)
        
        i_vals.append(i+1)
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
 
        acc_and_loss = [(i+1),temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x,3) for x in acc_and_loss]
        print("Iteration {}. Train Loss: {:.3f}. Train Acc: {:.3f}. Test Acc: {:.3f}".format(*acc_and_loss))
     
   
##     
     
plt.plot(i_vals, train_loss, 'k-')
plt.title("Softmax Loss for each iteration")
plt.xlabel("Iteration")
plt.ylabel("Softmax loss")
plt.show()


plt.plot(i_vals, train_acc, 'r-', label="Accuracy in training")
plt.plot(i_vals, test_acc, 'b--', label="Accuracy in testing")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim([0,100])
plt.title("Accuracy in prediction")
plt.legend(loc="lower right")
plt.show()


sample_idx = random.sample(range(len(images_test_28_gray)), 40)
sample_images1 = [images_test_28_gray[i] for i in sample_idx]
sample_images2 = np.expand_dims(sample_images1, 3)
sample_labels = [labels_test[i] for i in sample_idx]
prediction = sess.run([test_prediction], feed_dict={eval_input:sample_images2})[0]
prediction = [np.argmax(pred) for pred in prediction]

plt.figure(figsize=(16,20))
for i in range(len(sample_images1)):
    truth = sample_labels[i]
    predi = prediction[i]
    plt.subplot(10,4,i+1)
    plt.axis("off")
    color = "green" if truth==predi else "red"
    plt.text(32,15, "Real:         {0}\nPrediction:{1}".format(truth, predi),
            fontsize = 14, color = color)
    plt.imshow(sample_images1[i], cmap="gray")
plt.show()


## simple

ops.reset_default_graph()
sess = tf.compat.v1.Session()


## configuracion

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = images_train_28_gray[0].shape[0]
image_height = images_train_28_gray[0].shape[1]
target_size = max(labels_train)+1
num_chanels = 1
generations = 800
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
full_connected_size1 = 100

x_input_shape = (batch_size, image_width, image_height, num_chanels)
x_input = tf.compat.v1.placeholder(tf.float32, shape = x_input_shape)
y_target = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))

eval_input_shape = (None, image_width, image_height, num_chanels)
eval_input = tf.compat.v1.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.compat.v1.placeholder(tf.float32, shape = (evaluation_size))

conv1_weight = tf.Variable(tf.compat.v1.truncated_normal([4,4,num_chanels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
                    
conv2_weight = tf.Variable(tf.compat.v1.truncated_normal([4,4,conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

resulting_width = image_width // (max_pool_size1*max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)

full1_input_size = resulting_width*resulting_height*conv2_features
full1_weight = tf.Variable(tf.compat.v1.truncated_normal([full1_input_size, full_connected_size1], stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.compat.v1.truncated_normal([full_connected_size1], stddev=0.1, dtype = tf.float32))

full2_weight = tf.Variable(tf.compat.v1.truncated_normal([full_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.compat.v1.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def my_conv_neural_net(input_data):
    ## First layer: Conv+ReLU+Maxpool
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1,1,1,1], padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1,max_pool_size1, max_pool_size1,1], 
                               strides=[1, max_pool_size1, max_pool_size1,1], padding="SAME")
    ## Second layer: Conv+ReLU+Maxpool
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1,1,1,1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1,max_pool_size2, max_pool_size2,1], 
                               strides=[1, max_pool_size2, max_pool_size2,1], padding="SAME")
    ## Flattening
    flat_output = tf.compat.v1.layers.flatten(max_pool2)
    ## Third Layer: fully connected
    fully_connected_1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
    ## Fourth layer: fully connected
    fully_connected_2 = tf.add(tf.matmul(fully_connected_1, full2_weight), full2_bias)
    return fully_connected_2


model_ouput = my_conv_neural_net(x_input)
test_model_output = my_conv_neural_net(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_ouput, labels = y_target))

prediction = tf.nn.softmax(model_ouput)
test_prediction = tf.nn.softmax(test_model_output)

my_optim = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
train_step = my_optim.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

train_loss = []
train_acc = []
test_acc = []
i_vals = []
for i in range(generations):
    rand_idx = np.random.choice(len(images_train_28_gray), size = batch_size)
    rand_x = images_train_28_gray[rand_idx]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = labels_train[rand_idx]
    train_dict = {x_input:rand_x, y_target:rand_y}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)
    
    if(i+1) % eval_every == 0:
        rand_idx_eval = np.random.choice(len(images_test_28_gray), size = evaluation_size)
        rand_x_eval = images_test_28_gray[rand_idx_eval]
        rand_x_eval = np.expand_dims(rand_x_eval, 3)
        rand_y_eval = labels_test[rand_idx_eval]
        test_dict = {eval_input:rand_x_eval, eval_target:rand_y_eval}

        temp_test_preds = sess.run( test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(temp_test_preds, rand_y_eval)
        
        i_vals.append(i+1)
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
 
        acc_and_loss = [(i+1),temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x,3) for x in acc_and_loss]
        print("Iteration {}. Train Loss: {:.3f}. Train Acc: {:.3f}. Test Acc: {:.3f}".format(*acc_and_loss))
    
        
plt.plot(i_vals, train_loss, 'k-')
plt.title("Softmax loss for each iteration")
plt.xlabel("Iteration")
plt.ylabel("Softmax loss")
plt.show()


plt.plot(i_vals, train_loss, 'k-')
plt.title("Softmax loss for each iteration")
plt.xlabel("Iteration")
plt.ylabel("Softmax loss")
plt.show()

sample_idx = random.sample(range(len(images_test_28_gray)), 40)
sample_images1 = [images_test_28_gray[i] for i in sample_idx]
sample_images2 = np.expand_dims(sample_images1, 3)
sample_labels = [labels_test[i] for i in sample_idx]
prediction = sess.run([test_prediction], feed_dict={eval_input:sample_images2})[0]
prediction = [np.argmax(pred) for pred in prediction]

plt.figure(figsize=(16,20))
for i in range(len(sample_images1)):
    truth = sample_labels[i]
    predi = prediction[i]
    plt.subplot(10,4,i+1)
    plt.axis("off")
    color = "green" if truth==predi else "red"
    plt.text(32,15, "Real:         {0}\nPrediction:{1}".format(truth, predi),
            fontsize = 14, color = color)
    plt.imshow(sample_images1[i], cmap="gray")
plt.show()



## hasta aqui

