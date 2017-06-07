import os
import struct
import numpy as np
import random
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

with open('MNIST.dat','rb') as f:
    x = pickle.load(f)
train_dataset1 = x['train_dataset']
train_labels1 = x['train_labels']
test_dataset1 = x['test_dataset']
test_labels1 = x['test_labels']
del x

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 128
patch_size = 5
depth = 64
num_hidden = 400
image_size = 64
num_labels = 100
num_channels = 1 # grayscale

graph = tf.Graph()

with graph.as_default():

    with tf.device('/gpu:0'):
  # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_test_dataset = tf.constant(test_dataset1,dtype=tf.float32)
    

      # Variables.
      global_step = tf.Variable(0.0)
      layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.zeros([depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer3_biases = tf.Variable(tf.zeros([depth]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer4_biases = tf.Variable(tf.zeros([depth]))
      layer5_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer5_biases = tf.Variable(tf.zeros([depth]))
      layer6_weights = tf.Variable(tf.truncated_normal(
          [image_size //32 * image_size // 32*depth, num_hidden], stddev=0.1))
      layer6_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer7_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.01))
      layer7_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = conv = tf.nn.conv2d(hidden, layer3_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases)
	#hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')	
        conv = conv = tf.nn.conv2d(hidden, layer4_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer4_biases)
	#hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        conv = conv = tf.nn.conv2d(hidden, layer5_weights, [1, 2, 2, 1], padding='SAME')
	hidden = tf.nn.relu(conv + layer5_biases)
	#hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer6_weights) + layer6_biases)
        #hidden_dropout = tf.nn.dropout(hidden,0.5);
        return tf.matmul(hidden, layer7_weights) + layer7_biases

      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

      # Optimizer.
      learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.85, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss,global_step = global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      test_prediction = tf.nn.softmax(model(tf_test_dataset))
num_stepsList = [51,101,501,1001,1501,2001,2501,3001,3501,4001,4501,5001,5501,6001]
plt.ylabel('Test Accuracy')
plt.xlabel('No of steps')
num_steps = 5001
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		  offset = (step * batch_size) % (train_labels1.shape[0] - batch_size)
		  batch_data = train_dataset1[offset:(offset + batch_size), :, :, :]
		  batch_labels = train_labels1[offset:(offset + batch_size), :]
		  feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		  _, l, predictions = session.run(
		    [optimizer, loss, train_prediction], feed_dict=feed_dict)
		  if (step % 50 == 0):
		    print('Minibatch loss at step %d: %f' % (step, l))
		    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                     
                  if step in num_stepsList:
                  	acc = accuracy(test_prediction.eval(), test_labels1)
                  	plt.plot(step,acc,'ro')
                  
	print('Test accuracy: %.1f%%' %acc)
plt.show()
	
