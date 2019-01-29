mport numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import utils

#Defining the hyperparameters

learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000
n_hidden = 300

#Step 1: Read Data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten = True)

#Step 2: Create datasets and iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

#Create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

#Step 3: create weights and bias
w1 = tf.get_variable(name = 'weights_1', shape = (784, n_hidden), initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
b1 = tf.get_variable(name = 'bias_1', shape = (1, n_hidden), initializer = tf.zeros_initializer())

#Parameters for the hidden layer and the output layer
w2 = tf.get_variable(name = 'weights_2', shape = (n_hidden, 10), initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
b2 = tf.get_variable(name = 'bias_2', shape = (1, 10), initializer = tf.zeros_initializer())

#Step 4: build model
z1 = tf.matmul(img, w1) + b1
h1 = tf.nn.sigmoid(z1, name = 'sigmoid_layer')

logits = tf.matmul(h1, w2) + b2

#Step 5: define the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = label, logits = logits, name = 'entropy')
loss = tf.reduce_mean(entropy, name = 'loss')

#Step 6: define the training operation
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#Step 7: calculate the accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./Graph/logreg_improved', tf.get_default_graph())

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    #train the model number of epochs times
    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time : {0} seconds'.format(time.time() - start_time))

    #test model
    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/ n_test))
writer.close()
