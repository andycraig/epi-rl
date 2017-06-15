# Heavily based on https://gist.github.com/awjuliani/86ae316a231bceb96a3e2ab3ac8e646a#file-rl-tutorial-2-ipynb

import numpy as np
import tensorflow as tf
import math
import sys
import getopt
from utils import getAction, discount_rewards, output
import policyNetwork, valueNetwork


def main(argv):
	# hyperparameters
	learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
	gamma = 0.99 # discount factor for reward
	# hyperparameters that can be set with command line arguments.
	gridLength = 2
	graphics = False
	total_episodes = 10000
	timeRemaining = 5
	batch_size = 5 # every how many episodes to do a param update?
	beta = 0
	initiallyCryptic = False
	opts, args = getopt.getopt(argv,"h:n:t:b:g",
									["hostlength=","nepisodes=",
									"timeremaining=","batchsize=",
									"beta=","initiallyc"])
	for opt, arg in opts:
		if opt in ("-h", "--hostlength"):
			gridLength = int(arg)
		elif opt in ("-t", "--timeremaining"):
			timeRemaining = int(arg)
		elif opt in ("-n", "--nepisodes"):
			total_episodes = int(arg)
		elif opt in ("-b", "--batchsize"):
			batch_size = int(arg)
		elif opt in ("--beta"):
			beta = float(arg)
		elif opt in ("--initiallyc"):
			initiallyCryptic = True
	# Epidemic version.
	from epidemic import Epidemic
	env = Epidemic(gridLength=gridLength,
					epsilon=0,
					beta=beta,
					CToI=1,
					timeRemaining=timeRemaining,
					rewardForAnyNonI=False,
					initiallyCryptic=initiallyCryptic)
	D = env.nHosts
	nActions = env.nHosts + 1 # +1 for 'do nothing'.

	tf.reset_default_graph()
	# Define policy network placeholders.
	observations_placeholder = tf.placeholder(tf.float32, [None,D], name="input_x")
	input_y_placeholder = tf.placeholder(tf.float32,[None,D], name="input_y")
	advantages_placeholder = tf.placeholder(tf.float32,name="reward_signal")
	# Set up policy network.
	probability = policyNetwork.inference(observations_placeholder)
	loss =        policyNetwork.loss(probability, input_y_placeholder, advantages_placeholder) # Both?
	train_op =    policyNetwork.training(loss, learning_rate)
	# Set up value network placeholders.
	new_value_placeholder = tf.placeholder(tf.float32, [None,D], name="new_value_placeholder")
	# Set up value network.
	estimated_value = valueNetwork.inference(observations_placeholder)
	value_loss =      valueNetwork.loss(estimated_value, advantages_placeholder)
	value_train_op =  valueNetwork.training(value_loss, learning_rate)

	xs, ys = [],[]
	rewards, vals_from_network = [], []
	all_discounted_rewards = np.array([])
	all_discounted_vals_from_network = np.array([])
	episode_number = 0
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		observation = env.reset() # Obtain an initial observation of the environment

		while episode_number < total_episodes:
			episode_number += 1

			# Make sure the observation is in a shape the network can handle.
			x = np.reshape(observation,[1,D])

			# Run the policy network and get an action to take.
			# Purpose of action is soley to go into env.step().
			tfprob = sess.run(probability,
							feed_dict={observations_placeholder: x})
			this_val_from_network = sess.run(estimated_value,
							feed_dict={observations_placeholder: x})
			action, y = getAction(tfprob)

			xs.append(x) # observation
			ys.append(y)

			# step the environment and get new measurements
			observation, thisReward, done, info = env.step(action)
			rewards.append(thisReward) # record reward (has to be done after we call step() to get reward for previous action)
			# this_val_from_network is like [[something]].
			vals_from_network.append(this_val_from_network[0][0])

			if done:
				# Have to handle rewards differently from x, y, because they need to be
				# discounted and normalised on a per-episode basis.
				# compute the discounted reward backwards through time
				discounted_ep_rewards = discount_rewards(rewards, gamma=gamma)
				# TODO Should predicted vals be discounted?
				discounted_vals_from_network = discount_rewards(vals_from_network, gamma=gamma)
				# discounted_ep_rewards is numpy array.
				all_discounted_rewards = np.concatenate([all_discounted_rewards, discounted_ep_rewards])
				all_discounted_vals_from_network = np.concatenate([all_discounted_vals_from_network, discounted_vals_from_network])
				all_advantages = all_discounted_rewards - all_discounted_vals_from_network
				rewards, vals_from_network = [], []

				# If we have completed enough episodes, then update the policy network with our gradients.
				if episode_number % batch_size == 0:
					# Was: sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
					sess.run([train_op, loss],
						feed_dict={observations_placeholder: np.vstack(xs),
								input_y_placeholder: np.vstack(ys),
								advantages_placeholder: np.vstack(all_advantages)})
					_, thisValueLoss = sess.run([value_train_op, value_loss],
						feed_dict={observations_placeholder: np.vstack(xs),
								advantages_placeholder: np.vstack(all_discounted_rewards)})

					# Reset the arrays.
					xs, ys = [],[] # reset array memory
					rewards, vals_from_network = [], []
					all_discounted_rewards = []
					all_advantages = []
					all_discounted_vals_from_network = []
					# Give a summary of how well our network is doing for each batch of episodes.
					print('Ep %i/%i' % (episode_number, total_episodes))
					if (batch_size == 1) and ((episode_number == 1) or (episode_number == total_episodes)):
						print("Observation was: ")
						print(x)
						print("Probabilities were: ")
						print(tfprob)
						print("Action chosen was: ")
						if action == nActions:
							print(action, " (no action)")
						else:
							print(action)
						print("Reward was: ")
						print(thisReward)
						print("Value network thought value was: ")
						print(this_val_from_network[0][0])
						print("Value network loss was: ")
						print(thisValueLoss)

				observation = env.reset()

		print(episode_number,'Episodes completed.')
		# Run the trained model on a sample and save to a file.
		observation = env.reset()

		output(env, probability, observations_placeholder, "outputFile.txt", sess)

if __name__ == "__main__":
	  main(sys.argv[1:])
