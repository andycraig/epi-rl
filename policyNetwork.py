import tensorflow as tf
import numpy as np

def inference(observation, nActions, H): # 'Inference' in the sense of 'prediction'.
	W = []
	layers = []
	# From observations to hidden layer 0.
	with tf.variable_scope("policyNetwork"):
		W.append(tf.get_variable("W"+str(0), shape=[observation.get_shape()[1], H[0]],
					 initializer=tf.contrib.layers.xavier_initializer()))
		layers.append(tf.nn.relu(tf.matmul(observation,W[0])))
		# Intermediate layers.
		for iLayer in range(1, len(H)):
			W.append(tf.get_variable("W"+str(iLayer), shape=[H[iLayer-1], H[iLayer]],
						 initializer=tf.contrib.layers.xavier_initializer()))
			layers.append(tf.nn.relu(tf.matmul(layers[-1],W[-1])))
		# From last hidden layer to output.
		# Length of action options is same as length of observation, plus 1 for no action..
		W.append(tf.get_variable("W"+str(len(H)), shape=[H[-1], nActions],
					 initializer=tf.contrib.layers.xavier_initializer()))
		score = tf.matmul(layers[-1],W[-1])
		probability = tf.nn.softmax(score)
	return probability

def loss(probability, input_y, advantages):
	#From here we define the parts of the network needed for learning a good policy.

	# The loss function. This sends the weights in the direction of making actions
	# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
	# Modified version of original; this one has high likelihood when input_y and probability match up.
	# For example, if input_y is [0, 1] and probability is [.2, .8], we should get
	# tf.log()
	# For example, if input_y is [0, 1, 0] and probability is [.1, .8, .1], we should get
	# tf.log()
	loglik = tf.log(tf.reduce_sum(tf.mul(input_y, probability)))
	loss = -tf.reduce_mean(loglik * advantages)
	return loss
	#newGrads = tf.gradients(loss,tvars)

def training(loss, learning_rate):
	# Once we have collected a series of gradients from multiple episodes, we apply them.
	# We don't just apply gradients after every episode in order to account for noise in the reward signal.
	adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)# Our optimizer
	return adam # Probably wrong
