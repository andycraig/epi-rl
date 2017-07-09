# Heavily based on https://gist.github.com/awjuliani/86ae316a231bceb96a3e2ab3ac8e646a#file-rl-tutorial-2-ipynb

import gym
import numpy as np
import tensorflow as tf
import math
import sys
import getopt
import datetime
from utils import getAction, discount_rewards, output, softmax
import policyNetwork, valueNetwork


def main(argv):
	# hyperparameters
	learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
	gamma = 0.99 # discount factor for reward
	# hyperparameters that can be set with command line arguments.
	gridLength = 2
	graphics = False
	total_episodes = 1000
	timeRemaining = 1
	batch_size = 5 # every how many episodes to do a param update?
	beta = 0
	initiallyCryptic = False
	verbose = False
	layersPolicy = [5] # An element for each layer, with each element the number of nodes in the layer.
	layersValue = [5]
	useValueNetwork = False
	useAdvantageOracle = False
	environment = "epidemic"
	opts, args = getopt.getopt(argv,"e:h:n:t:b:g",
									["environment=","hostlength=","nepisodes=",
									"timeremaining=","batchsize=",
									"beta=","initiallyc","verbose","valuenetwork","cheat"])
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
		elif opt in ("-e", "--environment"):
			environment = arg
		elif opt in ("--initiallyc"):
			initiallyCryptic = True
		elif opt in ("--verbose"):
			verbose = True
		elif opt in ("--valuenetwork"):
			useValueNetwork = True
		elif opt in ("--cheat"):
			useAdvantageOracle = True
			timeRemaining = 1
	# Epidemic version.
	if environment == "cartpole":
		env = gym.make("CartPole-v0")
		D = 4
		nActions = 2
	elif environment == "epidemic":
		from epidemic import Epidemic
		env = Epidemic(gridLength=gridLength,
						epsilon=0,
						beta=beta,
						CToI=0,
						timeRemaining=timeRemaining,
						rewardForC=True,
						rewardForR=True,
						initiallyCryptic=initiallyCryptic)
		D = env.nHosts
		nActions = env.nHosts + 1 # +1 for 'do nothing'.
	else:
		raise ValueError("environment must be epidemic or cartpole.")

	tf.reset_default_graph()
	# Define policy network placeholders.
	observations_placeholder = tf.placeholder(tf.float32, [None,D], name="input_x")
	input_y_placeholder = tf.placeholder(tf.int32, [None], name="input_y")
	advantages_placeholder = tf.placeholder(tf.float32,name="reward_signal")
	# Set up policy network.
	logits = 	policyNetwork.inference(observations_placeholder, nActions, layersPolicy)
	loss =      policyNetwork.loss(logits, input_y_placeholder, advantages_placeholder, nActions) # Both?
	train_op =  policyNetwork.training(loss, learning_rate)
	if useValueNetwork:
		# Set up value network placeholders.
		new_value_placeholder = tf.placeholder(tf.float32, [None,D], name="new_value_placeholder")
		# Set up value network.
		estimated_value = valueNetwork.inference(observations_placeholder, layersValue)
		value_loss =      valueNetwork.loss(estimated_value, advantages_placeholder)
		value_train_op =  valueNetwork.training(value_loss, learning_rate)

	xs, ys = [],[]
	rewards = []
	all_discounted_rewards = np.array([])
	if useValueNetwork:
		vals_from_network = []
		all_discounted_vals_from_network = np.array([])

	episode_number = 1
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		# TensorBoard setup for reporting.
		summary_merged = tf.summary.merge_all()
		summaryDir = 'summary/' + str(datetime.datetime.now()).replace(":", "-")
		summary_writer = tf.summary.FileWriter(summaryDir, sess.graph)

		observation = env.reset() # Obtain an initial observation of the environment
		x = np.reshape(observation,[1,D])

		if verbose:
			tflogits = sess.run(logits,
							feed_dict={observations_placeholder: x})
			print("\n============= BEFORE TRAINING ============")
			print("Example observation: ", x)
			print("Logits before softmax were: ", tflogits)
			print("Probabilities for this were: ", softmax(tflogits[0]))

		while episode_number <= total_episodes:

			# Make sure the observation is in a shape the network can handle.
			x = np.reshape(observation,[1,D])

			# Run the policy network and get an action to take.
			# Purpose of action is soley to go into env.step().
			tflogits = sess.run(logits, feed_dict={observations_placeholder: x})
			if useValueNetwork:
				this_val_from_network = sess.run(estimated_value,
							feed_dict={observations_placeholder: x})
			action, y = getAction(tflogits)
			# step the environment and get new measurements
			observation, thisReward, done, info = env.step(action)

			xs.append(x) # observation
			ys.append(action)
			rewards.append(thisReward) # record reward (has to be done after we call step() to get reward for previous action)

			if useValueNetwork:
				# this_val_from_network is like [[something]].
				vals_from_network.append(this_val_from_network[0][0])

			if done:
				# Have to handle rewards differently from x, y, because they need to be
				# discounted and normalised on a per-episode basis.
				# compute the discounted reward backwards through time
				discounted_ep_rewards = discount_rewards(rewards, gamma=gamma)
				# TODO Should predicted vals be discounted?
				if useValueNetwork:
					discounted_vals_from_network = discount_rewards(vals_from_network, gamma=gamma)
				# discounted_ep_rewards is numpy array.
				all_discounted_rewards = np.concatenate([all_discounted_rewards, discounted_ep_rewards])
				if useValueNetwork:
					all_discounted_vals_from_network = np.concatenate([all_discounted_vals_from_network, discounted_vals_from_network])
					all_advantages = all_discounted_rewards - all_discounted_vals_from_network
				if verbose:
					print("\n=== END OF EPISODE ", episode_number, "/", total_episodes, " ===")
					print("ys: ", ys)
					print("all_discounted_rewards: ", all_discounted_rewards)
					if useValueNetwork:
						print("all_discounted_vals_from_network: ", all_discounted_vals_from_network)
						print("all_advantages: ", all_advantages)
				rewards = []
				if useValueNetwork:
					vals_from_network = []

				# If we have completed enough episodes, then update the policy network with our gradients.
				if episode_number % batch_size == 0:
					print("\n============= END OF BATCH ============")

					canLearn = True
					if not useValueNetwork:
						discountedRewardMean = np.mean(all_discounted_rewards)
						calculated_all_advantages = (all_discounted_rewards - discountedRewardMean)
						#discountedRewardStdev = np.sqrt(np.var(all_discounted_rewards))
						if np.var(all_discounted_rewards) == 0:
							Warning("Discounted rewards were all same for this batch - can't learn.")
							print("Discounted rewards were all same for this batch - can't learn.")
							canLearn = False
						if timeRemaining == 1:
							oracleDiscountedRewardMean = 1.0 / env.nHosts + (env.nHosts - 1)**2 / ((env.nHosts)**2)
							canLearn = True
							oracle_all_advantages = (all_discounted_rewards - oracleDiscountedRewardMean)
						if useAdvantageOracle:
							all_advantages = oracle_all_advantages
						else:
							all_advantages = calculated_all_advantages
					# Was: sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
					if canLearn:
						summary, thisPolicyTrain, thisPolicyLoss = sess.run([summary_merged, train_op, loss],
							feed_dict={observations_placeholder: np.vstack(xs),
									input_y_placeholder: ys,
									advantages_placeholder: all_advantages})
						if useValueNetwork:
							thisValueTrain, thisValueLoss = sess.run([value_train_op, value_loss],
								feed_dict={observations_placeholder: np.vstack(xs),
										advantages_placeholder: np.vstack(all_discounted_rewards)})
					# TensorBoard reporting.
					# Graph variables
					summary_writer.add_summary(summary, episode_number)
					# Non-graph variables.
					nongraph_summary = tf.Summary(value=[tf.Summary.Value(tag="discountedRewardMean", simple_value=discountedRewardMean)])
					summary_writer.add_summary(nongraph_summary, episode_number)
					# Console output if required.
					if verbose:
						print("All discounted rewards: ", all_discounted_rewards)
						print("Discounted reward mean: ", discountedRewardMean)
						print("Calculated advantages: ", calculated_all_advantages)
						print("Oracle advantages: ", oracle_all_advantages)
						print("Used advantages: ", all_advantages)
						print("Observations were: ", xs)
						print("Actions were: ", ys)
						if useValueNetwork:
							print("Estimated value was: ", this_val_from_network[0][0])
						tflogitsWouldBe = sess.run(logits,
							feed_dict={observations_placeholder: x})
						print("Logits before softmax were: ", tflogits)
						print("Now logits before softmax would be: ", tflogitsWouldBe)
						print("Probabilities for this were: ", softmax(tflogits[0]))
						print("Now probabilities would be: ", softmax(tflogitsWouldBe[0]))
					print("Average discounted reward in batch: ", np.mean(all_discounted_rewards))
					# Reset the arrays.
					xs, ys = [],[] # reset array memory
					rewards = []
					if useValueNetwork:
						vals_from_network = []
						all_discounted_vals_from_network = []
					all_discounted_rewards = []
					all_advantages = []
					# Give a summary of how well our network is doing for each batch of episodes.
				if not verbose:
					print('Ep %i/%i' % (episode_number, total_episodes))

				observation = env.reset()
				episode_number += 1

		print("==========", episode_number,'EPISODES COMPLETED ==========')

		observation = env.reset() # Obtain an initial observation of the environment
		x = np.reshape(observation,[1,D])
		print("Random observation: ", x)
		tflogits = sess.run(logits, feed_dict={observations_placeholder: x})
		print("Logits before softmax were: ", tflogits)
		print("Probabilities for this were: ", softmax(tflogits[0]))

if __name__ == "__main__":
	  main(sys.argv[1:])
