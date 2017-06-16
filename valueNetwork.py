import tensorflow as tf
import numpy as np

def inference(observation, H): # 'Inference' in the sense of 'prediction'.

	W = []
	layers = []
	with tf.variable_scope("valueNetwork"):
		# From observations to hidden layer 0.
		W.append(tf.get_variable("W"+str(0), shape=[observation.get_shape()[1], H[0]],
					 initializer=tf.contrib.layers.xavier_initializer()))
		layers.append(tf.nn.relu(tf.matmul(observation,W[0])))
		# Intermediate layers.
		for iLayer in range(1, len(H)):
			W.append(tf.get_variable("W"+str(iLayer), shape=[H[iLayer-1], H[iLayer]],
						 initializer=tf.contrib.layers.xavier_initializer()))
			layers.append(tf.matmul(layers[-1],W[-1]))
		# From last hidden layer to output.
		# Length of action options is same as length of observation.
		W.append(tf.get_variable("W"+str(len(H)), shape=[H[-1], 1],
					 initializer=tf.contrib.layers.xavier_initializer()))
		estimated_value = tf.matmul(layers[-1],W[-1])
	return estimated_value

def loss(estimated_value, new_value):
	loss = tf.nn.l2_loss(estimated_value - new_value)
	return loss

def training(loss, learning_rate):
	# Once we have collected a series of gradients from multiple episodes, we apply them.
	# We don't just apply gradients after every episode in order to account for noise in the reward signal.
	adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)# Our optimizer
	return adam # Probably wrong
