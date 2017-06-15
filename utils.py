import tensorflow as tf
import numpy as np

def discount_rewards(r, gamma):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def getAction(tfprob, env):
	# y is
	# tfprob and y should match up.
	# tfprob [1.] should give action = 0, y = [1]
	# tfprob [0.] should give action = 1, y = [0]
	# TODO Change from np.random.multinomial, which is very slow.
	pvals = tfprob[0]
	try:
		y = np.random.multinomial(n=1, pvals=pvals)
	except ValueError as e:
		# Sometimes if one probability is very large, numerical imprecision
		# results in sum(pvals) > 1 and there is a ValueError.
		# In this case, just take the largest value.
		action, pmax = max(enumerate(pvals), key=lambda x: x[1])
		if pmax < 0.75:
			print(pvals)
			raise ValueError("There was a ValueError, and largest prob was " + str(pmax) + " < 0.75.")
		else:
			print("Deterministic action to avoid error.")
			# y is one-hot of action.
			y = np.eye(len(pvals))[action,]
	if env == 'cartpole':
		action = 1 if (y == np.array([0,1])).all() else 0
	elif env == 'epidemic':
		try:
			action = np.asscalar(np.where(y == 1)[0])
		except ValueError:
			print("ValueError! y was ", y)
	return action, y

def output(env, scopeName, fileName, timeRemaining, sess):
	with tf.name_scope(scopeName):
		observation = env.reset()
		done = False
		with open(fileName, "w") as f:
			for iOutput in range(1):
				for iTime in range(timeRemaining):
					f.write(str(env))
					print(str(env))
					x = np.reshape(observation,[1,len(observation)])
					print(probability.name)
					tfprob = sess.run(probability,feed_dict={observations: x})
					f.write(str(np.reshape(tfprob[0][0:env.nHosts], [env.gridLength, env.gridLength])))
					print(str(np.reshape(tfprob[0][0:env.nHosts], [env.gridLength, env.gridLength])))
					f.write("Prob. of no action: " + str(tfprob[0][-1]))
					print("Prob. of no action: " + str(tfprob[0][-1]))
					action, y = getAction(tfprob, environment)
					f.write("Took action: " + str(action))
					print("Took action: " + str(action))
					observation, reward, done, info = env.step(action)
					f.write("Got reward: " + str(reward))
					print("Got reward: " + str(reward))
				observation = env.reset()
