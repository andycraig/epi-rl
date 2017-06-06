# Heavily based on https://gist.github.com/awjuliani/86ae316a231bceb96a3e2ab3ac8e646a#file-rl-tutorial-2-ipynb

import numpy as np
import tensorflow as tf
import math
import sys
import getopt

def getAction(tfprob, env):
	# tfprob and y should match up.
	# tfprob [1.] should give action = 0, y = [1]
	# tfprob [0.] should give action = 1, y = [0]
	# TODO Change from np.random.multinomial, which is very slow.
	pvals = tfprob[0]
	y = np.random.multinomial(n=1, pvals=pvals)
	if env == 'cartpole':
		action = 1 if (y == np.array([0,1])).all() else 0
	elif env == 'epidemic':
		try:
			action = np.asscalar(np.where(y == 1)[0])
		except ValueError:
			print("ValueError! y was ", y)
	return action, y

def main(argv):
	# hyperparameters
	H = [15] # number of hidden layer neurons
	batch_size = 5 # every how many episodes to do a param update?
	learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
	gamma = 0.99 # discount factor for reward
	# hyperparameters that can be set with command line arguments.
	environment = ''
	gridLength = None
	graphics = False
	total_episodes = 10000
	timeRemaining = 5
	try:
		opts, args = getopt.getopt(argv,"e:h:n:t:g",["env=","hostlength=","nepisodes=","timeremaining=","graphics="])
	except getopt.GetoptError:
		print('main.py -e <environment> -g')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-e", "--env"):
			environment = arg
		elif opt in ("-h", "--hostlength"):
			gridLength = int(arg)
		elif opt in ("-t", "--timeremaining"):
			timeRemaining = int(arg)
		elif opt in ("-n", "--nepisodes"):
			total_episodes = int(arg)
		elif opt in ("-g", "--graphics"):
			graphics = True
	if environment == 'epidemic':
		# Epidemic version.
		from epidemic import Epidemic
		env = Epidemic(gridLength=gridLength,
						epsilon=0,
						beta=0,
						CToI=1,
						timeRemaining=timeRemaining,
						rewardForAnyNonI=True)
		D = env.nHosts
		nActions = env.nHosts + 1 # +1 for 'do nothing'.
	elif environment == 'cartpole':
		# Cartpole version.
		import gym
		env = gym.make('CartPole-v0')
		D = 4 # input dimensionality
		nActions = 2
		if gridLength != None:
			print("Ignoring hostlength argument for cartpole environment.")
	else:
		raise ValueError("--env must be epidemic or cartpole.")



	tf.reset_default_graph()

	#This defines the network as it goes from taking an observation of the environment to
	#giving a prob of chosing to the action of moving left or right.
	W = []
	layers = []
	W1String = "W1"
	W2String = "W2"
	observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
	# From observations to hidden layer 0.
	W.append(tf.get_variable("W0", shape=[D, H[0]],
			   initializer=tf.contrib.layers.xavier_initializer()))
	layers.append(tf.nn.relu(tf.matmul(observations,W[0])))
	# Intermediate layers.
	for iLayer in range(1, len(H)):
		W.append(tf.get_variable("W"+str(iLayer), shape=[H[iLayer-1], H[iLayer]],
				   initializer=tf.contrib.layers.xavier_initializer()))
		layers.append(tf.nn.relu(tf.matmul(observations,W[iLayer-1])))
	# From last hidden layer to output.
	W.append(tf.get_variable("W"+str(len(H)), shape=[H[-1], nActions],
			   initializer=tf.contrib.layers.xavier_initializer()))
	score = tf.matmul(layers[-1],W[1])
	probability = tf.nn.softmax(score)

	#From here we define the parts of the network needed for learning a good policy.
	tvars = tf.trainable_variables()
	input_y = tf.placeholder(tf.float32,[None,nActions], name="input_y")
	advantages = tf.placeholder(tf.float32,name="reward_signal")

	# The loss function. This sends the weights in the direction of making actions
	# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
	# Modified version of original; this one has high likelihood when input_y and probability match up.
	# For example, if input_y is [0, 1] and probability is [.2, .8], we should get
	# tf.log()
	# For example, if input_y is [0, 1, 0] and probability is [.1, .8, .1], we should get
	# tf.log()
	loglik = tf.log(tf.reduce_sum(tf.mul(input_y, probability)))
	loss = -tf.reduce_mean(loglik * advantages)
	newGrads = tf.gradients(loss,tvars)

	# Once we have collected a series of gradients from multiple episodes, we apply them.
	# We don't just apply gradients after every episode in order to account for noise in the reward signal.
	adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
	W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
	W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
	batchGrad = [W1Grad,W2Grad]
	updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

	#Advantage function
	#This function allows us to weigh the rewards our agent recieves. In the context of the Cart-Pole task, we want actions that kept the pole in the air a long time to have a large reward, and actions that contributed to the pole falling to have a decreased or negative reward. We do this by weighing the rewards from the end of the episode, with actions at the end being seen as negative, since they likely contributed to the pole falling, and the episode ending. Likewise, early actions are seen as more positive, since they weren't responsible for the pole falling.

	def discount_rewards(r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * gamma + r[t]
			discounted_r[t] = running_add
		return discounted_r

	#Running the Agent and Environment
	#Here we run the neural network agent, and have it act in the CartPole environment.

	xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
	running_reward = None
	reward_sum = 0
	episode_number = 1
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		rendering = False
		sess.run(init)
		# Get the initial variable values, so we can compare them to the final values.
		W1Init = sess.run(W[0])
		W2Init = sess.run(W[1])

		# Run the trained model on a sample and save to a file.
		observation = env.reset()
		done = False
		with open("probsInitial.txt", "w") as f:
			f.write(str(env))
			x = np.reshape(observation,[1,D])
			tfprob = sess.run(probability,feed_dict={observations: x})
			f.write(str(np.reshape(tfprob[0][0:env.nHosts], [env.gridLength, env.gridLength])))
			f.write("\nProb. of no action: " + str(tfprob[0][-1]))
		with open("outputInitial.txt", "w") as f:
			while not done:
				f.write(str(env))
				x = np.reshape(observation,[1,D])
				tfprob = sess.run(probability,feed_dict={observations: x})
				action, y = getAction(tfprob, environment)
				observation, reward, done, info = env.step(action)
			# Output final state.
			f.write(str(env))
			f.write("Reward: " + str(reward))

		observation = env.reset() # Obtain an initial observation of the environment

		# Reset the gradient placeholder. We will collect gradients in
		# gradBuffer until we are ready to update our policy network.
		gradBuffer = sess.run(tvars)
		for ix,grad in enumerate(gradBuffer):
			gradBuffer[ix] = grad * 0

		while episode_number <= total_episodes:

			# Rendering the environment slows things down,
			# so let's only look at it once our agent is doing a good job.
			if reward_sum/batch_size > 100 or rendering == True :
				if graphics:
					env.render()
				rendering = True

			# Make sure the observation is in a shape the network can handle.
			x = np.reshape(observation,[1,D])

			# Run the policy network and get an action to take.
			# Purpose of action is soley to go into env.step().
			tfprob = sess.run(probability,feed_dict={observations: x})
			action, y = getAction(tfprob, environment)

			xs.append(x) # observation
			ys.append(y)

			# step the environment and get new measurements
			observation, reward, done, info = env.step(action)

			reward_sum += reward

			drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

			if done:
				episode_number += 1
				# stack together all inputs, hidden states, action gradients, and rewards for this episode
				epx = np.vstack(xs)
				epy = np.vstack(ys)
				epr = np.vstack(drs)
				tfp = tfps
				xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

				# compute the discounted reward backwards through time
				discounted_epr = discount_rewards(epr)
				# size the rewards to be unit normal (helps control the gradient estimator variance)
				#discounted_epr -= np.mean(discounted_epr)
				#TODO Next bit fails if no reward. Scaling is probably unnecessary if there is no reward?
				#if np.std(discounted_epr) > 0:
				#	discounted_epr /= np.std(discounted_epr)

				# Get the gradient for this episode, and save it in the gradBuffer
				tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
				for ix,grad in enumerate(tGrad):
					gradBuffer[ix] += grad

				# If we have completed enough episodes, then update the policy network with our gradients.
				if episode_number % batch_size == 0:
					sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
					for ix,grad in enumerate(gradBuffer):
						gradBuffer[ix] = grad * 0

					# Give a summary of how well our network is doing for each batch of episodes.
					running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
					print('Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size))

					reward_sum = 0

				observation = env.reset()

		print(episode_number,'Episodes completed.')
		# Get the final variable values.
		W1Final = sess.run(W[0])
		W2Final = sess.run(W[1])
		#print("Initial value of ", W1String,":\n",W1Init)
		#print("Initial value of ", W2String,":\n",W2Init)
		#print("Final value of ", W1String,":\n",W1Final)
		#print("Final value of ", W2String,":\n",W2Final)
		# Run the trained model on a sample and save to a file.
		observation = env.reset()
		done = False
		with open("probsFinal.txt", "w") as f:
			f.write(str(env))
			print(str(env))
			x = np.reshape(observation,[1,D])
			tfprob = sess.run(probability,feed_dict={observations: x})
			f.write(str(np.reshape(tfprob[0][0:env.nHosts], [env.gridLength, env.gridLength])))
			print(str(np.reshape(tfprob[0][0:env.nHosts], [env.gridLength, env.gridLength])))
			f.write("\nProb. of no action: " + str(tfprob[0][-1]))
			print("\nProb. of no action: " + str(tfprob[0][-1]))
		with open("outputFinal.txt", "w") as f:
			while not done:
				f.write(str(env))
				x = np.reshape(observation,[1,D])
				tfprob = sess.run(probability,feed_dict={observations: x})
				action, y = getAction(tfprob, environment)
				observation, reward, done, info = env.step(action)
			# Output final state.
			f.write(str(env))
			f.write("Reward: " + str(reward))


if __name__ == "__main__":
	  main(sys.argv[1:])
